import ast
import json
import keyword
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
import warnings

import chromadb
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# Configuration
BASE_DIR = "f:/Thesis_methodology"
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_FILE = os.path.join(BASE_DIR, "minGPT", "mingpt", "model.py")

ABLATION_PROFILES = {
    "phi_only": {
        "label": "Phi only",
        "use_ast": False,
        "use_intent": False,
        "context_sources": [],
    },
    "phi_ast": {
        "label": "Phi + AST",
        "use_ast": True,
        "use_intent": False,
        "context_sources": [],
    },
    "phi_ast_intent": {
        "label": "Phi + AST + Intent",
        "use_ast": True,
        "use_intent": True,
        "context_sources": [],
    },
    "phi_ast_intent_repository": {
        "label": "Phi + AST + Intent + Repository",
        "use_ast": True,
        "use_intent": True,
        "context_sources": ["repository"],
    },
    "phi_ast_intent_external_docs": {
        "label": "Phi + AST + Intent + External Docs",
        "use_ast": True,
        "use_intent": True,
        "context_sources": ["external_docs"],
    },
    "phi_ast_intent_repository_external_docs": {
        "label": "Phi + AST + Intent + Repository + External Docs",
        "use_ast": True,
        "use_intent": True,
        "context_sources": ["repository", "external_docs"],
    },
    "full_rag": {
        "label": "Full RAG",
        "use_ast": True,
        "use_intent": True,
        "context_sources": ["repository", "external_docs"],
    },
}

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "when", "where",
    "which", "while", "will", "have", "has", "been", "are", "was", "were", "than",
    "then", "else", "does", "using", "used", "value", "values", "return", "returns",
    "parameter", "parameters", "function", "method", "class", "code", "comment",
    "docstring", "input", "output", "data", "type", "types"
}


def load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_ollama_base_url():
    load_env_file()
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ollama_model():
    load_env_file()
    return os.environ.get("OLLAMA_MODEL", "phi4-mini")


def call_ollama(prompt):
    base_url = get_ollama_base_url().rstrip("/")
    model = get_ollama_model()
    request_body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_message = error.reason
        try:
            error_payload = json.loads(error.read().decode("utf-8"))
            error_message = error_payload.get("error", error_message)
        except Exception:
            pass
        raise RuntimeError(f"Ollama request failed: {error_message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. Make sure Ollama is running and the model '{model}' is available."
        ) from error

    return payload["message"]["content"]


def get_function_source(file_path, class_name, func_name):
    with open(file_path, "r", encoding="utf-8") as file_handle:
        tree = ast.parse(file_handle.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef) and subnode.name == func_name:
                    return ast.get_source_segment(open(file_path).read(), subnode), subnode
    return None, None


def flatten_ast(node):
    """
    Returns a string representation of the AST node.
    In Phase 4, this provides structural context to the LLM.
    """
    return ast.dump(node)


def get_root_definition_node(tree):
    if len(tree.body) != 1:
        return None

    root = tree.body[0]
    if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return root

    return None


def compute_line_offsets(text):
    offsets = [0]
    running_total = 0
    for line in text.splitlines(keepends=True):
        running_total += len(line)
        offsets.append(running_total)
    return offsets


def relative_position_to_offset(text, line, character):
    offsets = compute_line_offsets(text)
    if line >= len(offsets):
        return len(text)
    return offsets[line] + character


def remove_relative_range(text, range_info):
    start = relative_position_to_offset(text, range_info["startLine"], range_info["startCharacter"])
    end = relative_position_to_offset(text, range_info["endLine"], range_info["endCharacter"])
    return text[:start] + text[end:]


def clean_comment_text(text):
    lines = []
    for raw_line in text.strip().splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        lines.append(stripped)
    return "\n".join(lines).strip()


def extract_docstring_info(tree):
    root = get_root_definition_node(tree)
    if not root or not root.body:
        return None

    first_statement = root.body[0]
    if not (
        isinstance(first_statement, ast.Expr)
        and isinstance(getattr(first_statement, "value", None), ast.Constant)
        and isinstance(first_statement.value.value, str)
    ):
        return None

    doc_text = ast.get_docstring(root, clean=False) or ""
    indent = " " * first_statement.col_offset
    return {
        "kind": "docstring",
        "text": doc_text,
        "indent": indent,
        "replaceRange": {
            "startLine": first_statement.lineno - 1,
            "startCharacter": first_statement.col_offset,
            "endLine": first_statement.end_lineno - 1,
            "endCharacter": first_statement.end_col_offset,
        },
    }


def extract_inline_comment_info(code_text, tree):
    root = get_root_definition_node(tree)
    if not root or not root.body:
        return None

    lines = code_text.splitlines()
    first_statement = root.body[0]
    search_start = root.lineno
    search_end = first_statement.lineno - 1
    if search_end < search_start:
        return None

    comment_start = None
    comment_end = None
    comment_lines = []

    for index in range(search_start, search_end + 1):
        if index >= len(lines):
            break

        line = lines[index]
        stripped = line.strip()
        if not stripped:
            if comment_start is not None:
                comment_end = index
            continue

        if stripped.startswith("#"):
            if comment_start is None:
                comment_start = index
            comment_end = index
            comment_lines.append(stripped)
            continue

        if comment_start is not None:
            break

    if comment_start is None:
        return None

    start_character = len(lines[comment_start]) - len(lines[comment_start].lstrip())
    end_line = comment_end
    end_character = len(lines[end_line])
    indent = " " * start_character

    return {
        "kind": "line_comment",
        "text": clean_comment_text("\n".join(lines[comment_start:comment_end + 1])),
        "indent": indent,
        "replaceRange": {
            "startLine": comment_start,
            "startCharacter": start_character,
            "endLine": end_line,
            "endCharacter": end_character,
        },
    }


def extract_leading_comment_info(code_text):
    lines = code_text.splitlines()
    comment_start = None
    comment_end = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if comment_start is not None:
                comment_end = index
            continue

        if stripped.startswith("#"):
            if comment_start is None:
                comment_start = index
            comment_end = index
            continue

        break

    if comment_start is None:
        return None

    start_character = len(lines[comment_start]) - len(lines[comment_start].lstrip())
    end_character = len(lines[comment_end])
    indent = " " * start_character

    return {
        "kind": "line_comment",
        "text": clean_comment_text("\n".join(lines[comment_start:comment_end + 1])),
        "indent": indent,
        "replaceRange": {
            "startLine": comment_start,
            "startCharacter": start_character,
            "endLine": comment_end,
            "endCharacter": end_character,
        },
    }


def determine_insert_position(code_text, tree):
    root = get_root_definition_node(tree)
    if not root or not root.body:
        return {
            "line": 0,
            "character": 0,
            "indent": "",
        }

    for statement in root.body:
        if hasattr(statement, "lineno") and hasattr(statement, "col_offset"):
            return {
                "line": statement.lineno - 1,
                "character": statement.col_offset,
                "indent": " " * statement.col_offset,
            }

    indent_size = root.col_offset + 4
    return {
        "line": root.lineno,
        "character": indent_size,
        "indent": " " * indent_size,
    }


def extract_documentation_info(code_text, tree):
    doc_info = extract_docstring_info(tree)
    if doc_info:
        return doc_info

    inline_comment_info = extract_inline_comment_info(code_text, tree)
    if inline_comment_info:
        return inline_comment_info

    leading_comment_info = extract_leading_comment_info(code_text)
    if leading_comment_info:
        return leading_comment_info

    return None


def tokenize_meaningful_words(text):
    tokens = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower()):
        if token in STOPWORDS or token in keyword.kwlist or len(token) < 3:
            continue
        tokens.add(token)
    return tokens


def lexical_overlap_score(left_text, right_text):
    left_tokens = tokenize_meaningful_words(left_text)
    right_tokens = tokenize_meaningful_words(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def count_syllables(word):
    lowered = word.lower()
    groups = re.findall(r"[aeiouy]+", lowered)
    count = len(groups)
    if lowered.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def compute_readability_metrics(text):
    words = re.findall(r"[A-Za-z]+", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [sentence for sentence in sentences if sentence.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    if word_count == 0:
        return {
            "word_count": 0,
            "flesch_reading_ease": 0.0,
            "gunning_fog_index": 100.0,
        }

    syllable_count = sum(count_syllables(word) for word in words)
    complex_words = sum(1 for word in words if count_syllables(word) >= 3)

    flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
    gunning_fog_index = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))

    return {
        "word_count": word_count,
        "flesch_reading_ease": flesch_reading_ease,
        "gunning_fog_index": gunning_fog_index,
    }


def classify_comment_state(existing_comment, code_text, intent):
    if not existing_comment:
        return {
            "mode": "generate",
            "reason": "No existing documentation was found for the selected block.",
        }

    alignment = max(
        lexical_overlap_score(existing_comment, code_text),
        lexical_overlap_score(existing_comment, intent),
    )
    readability = compute_readability_metrics(existing_comment)

    if alignment < 0.08:
        return {
            "mode": "refactor",
            "reason": f"Existing documentation appears semantically misaligned (alignment={alignment:.2f}).",
            "metrics": readability,
        }

    if (
        readability["word_count"] < 20
        or readability["flesch_reading_ease"] < 30
        or readability["gunning_fog_index"] > 18
    ):
        return {
            "mode": "elaborate",
            "reason": (
                "Existing documentation is consistent enough but too short or hard to read "
                f"(words={readability['word_count']}, FRE={readability['flesch_reading_ease']:.1f}, "
                f"GFI={readability['gunning_fog_index']:.1f})."
            ),
            "metrics": readability,
        }

    return {
        "mode": "retain",
        "reason": "Existing documentation is already aligned and sufficiently descriptive.",
        "metrics": readability,
    }


def strip_markdown_fences(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def extract_docstring_block(text):
    match = re.search(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', text)
    if match:
        return match.group(1).strip()
    return None


def normalize_docstring_text(text):
    stripped = textwrap.dedent(strip_markdown_fences(text)).strip()
    extracted_docstring = extract_docstring_block(stripped)
    if extracted_docstring:
        return extracted_docstring
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return stripped
    if "\n" in stripped:
        return f'"""\n{stripped}\n"""'
    return f'"""{stripped}"""'


def indent_block(text, indent, indent_first_line=True):
    lines = text.splitlines()
    if not lines:
        return text

    if indent_first_line:
        formatted_lines = [(indent + lines[0]) if lines[0] else indent]
    else:
        formatted_lines = [lines[0]]

    for line in lines[1:]:
        formatted_lines.append((indent + line) if line else indent)

    return "\n".join(formatted_lines)


def build_mode_specific_prompt(mode, existing_comment):
    if mode == "generate":
        return """
Generate a fresh NumPy-style Python docstring because the selected code has no usable documentation.
"""

    if mode == "refactor":
        return f"""
Refactor the existing documentation because it does not align well with the selected code.
Existing documentation:
{existing_comment}

Preserve only ideas that remain semantically correct. Rewrite anything inaccurate.
"""

    if mode == "elaborate":
        return f"""
Elaborate the existing documentation because it is too short or hard to read.
Existing documentation:
{existing_comment}

Keep the correct meaning, but expand it into a clearer NumPy-style Python docstring.
"""

    return f"""
Keep the intent of the existing documentation, but if you improve wording, stay semantically equivalent.
Existing documentation:
{existing_comment}
"""


def find_intent(code):
    """
    Phase 3: Internal Context Retrieval.
    Summarize the logical intent of the code block while preserving
    the expected "Intent" and "Logic" structure.
    """
    prompt = f"""
You are analyzing a selected code snippet.

Return exactly two labeled lines in this structure:
Intent: <one concise sentence describing the function's purpose>
Logic: <one concise sentence describing the main implementation steps>

Do not add markdown, bullets, headings, or extra text.

Code:
```python
{code}
```
"""
    response_text = call_ollama(prompt).strip()

    if "Intent:" not in response_text or "Logic:" not in response_text:
        raise RuntimeError(
            "Ollama response for intent did not match the required 'Intent'/'Logic' structure."
        )

    return response_text


def normalize_ablation_profile(ablation_profile):
    if ablation_profile is None:
        return ABLATION_PROFILES["full_rag"]

    if isinstance(ablation_profile, str):
        if ablation_profile not in ABLATION_PROFILES:
            available_profiles = ", ".join(sorted(ABLATION_PROFILES))
            raise ValueError(f"Unknown ablation profile '{ablation_profile}'. Available profiles: {available_profiles}")
        return ABLATION_PROFILES[ablation_profile]

    return ablation_profile


def source_matches_context(source, context_sources):
    source = source or ""
    if "repository" in context_sources:
        if source.startswith("README") or source.startswith("Issue/PR"):
            return True
    if "external_docs" in context_sources:
        if source.startswith("PyPI"):
            return True
    return False


def retrieve_context(query_text, context_sources=None, k=3):
    """
    Phase 2 & 4: Query the Vector Database for documentation context.
    """
    context_sources = context_sources or ["repository", "external_docs"]
    print("Connecting to ChromaDB...", file=sys.stderr)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="repo_context")

    print("Loading embedding model for retrieval...", file=sys.stderr)
    model = SentenceTransformer(EMBED_MODEL)

    query_embed = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embed,
        n_results=12,
        include=["documents", "metadatas"],
    )

    filtered_documents = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        if source_matches_context(metadata.get("source"), context_sources):
            filtered_documents.append(document)
        if len(filtered_documents) == k:
            break

    return filtered_documents


def construct_unified_prompt(code, flattened_ast=None, rag_context=None, intent=None, profile_label="Full RAG"):
    """
    Phase 4: Construct the prompt for the Generation model.
    """
    sections = [
        "## TARGET FUNCTION GENERATION PROMPT",
        f"## ABLATION PROFILE: {profile_label}",
        "",
        "### 1. RAW LEXICAL CODE",
        "```python",
        code,
        "```",
    ]

    if flattened_ast:
        sections.extend([
            "",
            "### 2. FLATTENED AST STRING",
            flattened_ast,
        ])

    if rag_context:
        sections.extend([
            "",
            "### 3. RETRIEVED RAG CONTEXT",
            "\n".join(["- " + context_item for context_item in rag_context]),
        ])

    if intent:
        sections.extend([
            "",
            "### 4. DISCOVERED INTENT",
            intent,
        ])

    sections.extend([
        "",
        "### TASK",
        "Generate a highly accurate, professional NumPy-style docstring comment for the function above.",
        "Use only the evidence included in this ablation profile.",
        "",
        "### OUTPUT",
    ])

    return "\n".join(sections)


def construct_legacy_unified_prompt(code, flattened_ast, rag_context, intent):
    return f"""
## TARGET FUNCTION GENERATION PROMPT

### 1. RAW LEXICAL CODE
```python
{code}
```

### 2. FLATTENED AST STRING
{flattened_ast}

### 3. RETRIEVED VERSIONED RAG CONTEXT
{chr(10).join(['- ' + c for c in rag_context])}

### 4. DISCOVERED INTENT (PHASE 3)
{intent}

### TASK
Generate a highly accurate, professional NumPy-style docstring comment for the function above,
reflecting the specific mathematical intent and framework versions found in the context.

### OUTPUT
"""


def build_edit_payload(mode, comment_text, documentation_info, insert_position):
    if documentation_info and documentation_info.get("replaceRange"):
        replace_range = documentation_info["replaceRange"]
        return {
            "type": "replace",
            "startLine": replace_range["startLine"],
            "startCharacter": replace_range["startCharacter"],
            "endLine": replace_range["endLine"],
            "endCharacter": replace_range["endCharacter"],
            "text": comment_text,
        }

    return {
        "type": "insert",
        "startLine": insert_position["line"],
        "startCharacter": 0,
        "endLine": insert_position["line"],
        "endCharacter": 0,
        "text": comment_text + "\n",
    }


def generate_comment_from_code(code_text, ablation_profile=None, k=3):
    profile = normalize_ablation_profile(ablation_profile)
    tree = ast.parse(code_text)
    documentation_info = extract_documentation_info(code_text, tree)
    insert_position = determine_insert_position(code_text, tree)

    analysis_code = code_text
    existing_comment = ""
    if documentation_info:
        existing_comment = documentation_info["text"]
        analysis_code = remove_relative_range(code_text, documentation_info["replaceRange"]).strip() or code_text

    intent = ""
    if profile["use_intent"]:
        intent = find_intent(analysis_code)

    ast_str = ""
    if profile["use_ast"]:
        ast_tree = ast.parse(analysis_code)
        ast_str = flatten_ast(ast_tree)

    context = []
    if profile["context_sources"]:
        try:
            context = retrieve_context(analysis_code, profile["context_sources"], k=k)
        except Exception:
            context = ["No context found"]
        if not context:
            context = ["No context found for selected ablation sources."]

    classification = classify_comment_state(existing_comment, analysis_code, intent or analysis_code)

    if classification["mode"] == "retain":
        return {
            "ablation_profile": profile["label"],
            "mode": "retain",
            "message": classification["reason"],
            "comment": existing_comment,
            "edit": {
                "type": "none",
                "startLine": 0,
                "startCharacter": 0,
                "endLine": 0,
                "endCharacter": 0,
                "text": "",
            },
        }

    prompt = construct_unified_prompt(
        analysis_code,
        flattened_ast=ast_str,
        rag_context=context,
        intent=intent,
        profile_label=profile["label"],
    )
    mode_prompt = build_mode_specific_prompt(classification["mode"], existing_comment)
    raw_docstring = call_ollama(
        f"""
{prompt}

{mode_prompt}

Return only the final generated Python docstring.
Do not repeat the function signature, decorator, class header, or any code line.
Do not wrap the docstring in another quoted string.
The response must start with triple quotes and end with triple quotes.
Do not include explanations, markdown fences, or any surrounding text.
The output must be valid docstring content that can be inserted into the selected code block.
"""
    ).strip()

    normalized_docstring = normalize_docstring_text(raw_docstring)
    indent = documentation_info["indent"] if documentation_info else insert_position["indent"]
    formatted_docstring = indent_block(
        normalized_docstring,
        indent,
        indent_first_line=not bool(documentation_info),
    )
    edit = build_edit_payload(classification["mode"], formatted_docstring, documentation_info, insert_position)

    return {
        "ablation_profile": profile["label"],
        "mode": classification["mode"],
        "message": classification["reason"],
        "comment": formatted_docstring,
        "edit": edit,
    }


if __name__ == "__main__":
    try:
        code_input = sys.stdin.read()
        result = generate_comment_from_code(code_input)
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
