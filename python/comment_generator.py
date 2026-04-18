import ast
import os
import sys
import json
import urllib.request
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = "f:/Thesis_methodology"
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_FILE = os.path.join(BASE_DIR, "minGPT", "mingpt", "model.py")


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


def get_openai_api_key():
    load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def call_openai(prompt):
    api_key = get_openai_api_key()
    request_body = {
        "model": "gpt-5.4-mini",
        "input": prompt
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload["output"][0]["content"][0]["text"]


def get_function_source(file_path, class_name, func_name):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
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
    response_text = call_openai(prompt).strip()

    if "Intent:" not in response_text or "Logic:" not in response_text:
        raise RuntimeError(
            "OpenAI response for intent did not match the required 'Intent'/'Logic' structure."
        )

    return response_text


def retrieve_context(query_text):
    """
    Phase 2 & 4: Query the Vector Database for documentation context.
    """
    print("Connecting to ChromaDB...", file=sys.stderr)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="repo_context")
    
    print("Loading embedding model for retrieval...", file=sys.stderr)
    model = SentenceTransformer(EMBED_MODEL)
    
    query_embed = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embed,
        n_results=3
    )
    return results['documents'][0]
    # docs = results["documents"][0]
    # metas = results["metadatas"][0]

    # return [
    #     f"[{m.get('source','unknown')}] {d}"
    #     for d, m in zip(docs, metas)
    # ]


def construct_unified_prompt(code, flattened_ast, rag_context, intent):
    """
    Phase 4: Construct the prompt for the Generation model.
    """
    prompt = f"""
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
    print(prompt, file=sys.stderr)
    return prompt



def generate_comment_from_code(code_text):
    openai_api_key = get_openai_api_key()

    intent = find_intent(code_text)
    ast_tree = ast.parse(code_text)
    ast_str = flatten_ast(ast_tree)

    try:
        context = retrieve_context(code_text)
    except Exception:
        context = ["No context found"]

    prompt = construct_unified_prompt(code_text, ast_str, context, intent)

    # 👉 HERE you would call LLM (replace with real API)
    generated_comment = call_openai(
        f"""
{prompt}

Return only the final generated Python docstring.
Do not include explanations, markdown fences, or any surrounding text.
The output must be valid docstring content that can be inserted above the selected code.
"""
    ).strip()

    return generated_comment


# if __name__ == "__main__":
#     class_name = "CausalSelfAttention"
#     func_name = "forward"
    
#     print(f"--- RAG-AUGMENTED COMMENT GENERATION WORKFLOW ---")
#     print(f"Step 1: Extracting code for {class_name}.{func_name}...")
#     code, node = get_function_source(TARGET_FILE, class_name, func_name)
    
#     if code:
#         print("Step 2: Generating flattened AST...")
#         ast_str = flatten_ast(node)
        
#         print("Step 3: Discovering function intent (Phase 3 simulation)...")
#         intent = find_intent(code)
        
#         print("Step 4: Retrieving documentation context from ChromaDB (Phase 4 retrieval)...")
#         try:
#             context = retrieve_context(code)
#         except Exception as e:
#             print(f"Error during retrieval: {e}")
#             context = ["[Error: ChromaDB not initialized or empty. Run knowledge_base_construction.py first.]"]
        
#         print("Step 5: Constructing Unified Prompt...")
#         unified_prompt = construct_unified_prompt(code, ast_str, context, intent)
        
#         output_path = os.path.join(BASE_DIR, "final_generation_prompt.txt")
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(unified_prompt)
            
#         print(f"\nSUCCESS: Unified prompt generated and saved to {output_path}")
#         print("\n--- PREVIEW OF CONSTRUCTED PROMPT ---")
#         # Print first 500 chars for preview
#         print(unified_prompt[:1000] + "...")
#     else:
#         print("Function not found in target file.")

if __name__ == "__main__":
    try:
        code_input = sys.stdin.read()
        result = generate_comment_from_code(code_input)
        print(json.dumps({"comment": result}))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
