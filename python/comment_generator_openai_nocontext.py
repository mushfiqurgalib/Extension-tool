import json
import os
import re
import sys
import textwrap
import urllib.request


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


def get_openai_model():
    load_env_file()
    return os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


def call_openai(prompt):
    api_key = get_openai_api_key()
    request_body = {
        "model": get_openai_model(),
        "input": prompt,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload["output"][0]["content"][0]["text"]


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


def build_prompt(code_text):
    return f"""
### RAW LEXICAL CODE
```python
{code_text}
```

### TASK
Generate a highly accurate, professional NumPy-style docstring comment for the function above.

### OUTPUT
Return only the final Python docstring.
Do not repeat the function signature, decorator, class header, or any code line.
Do not wrap the docstring in another quoted string.
The response must start with triple quotes and end with triple quotes.
Do not include explanations, markdown fences, or any surrounding text.
"""


def generate_comment_from_code(code_text):
    prompt = build_prompt(code_text)
    print("\n===== OPENAI NO-CONTEXT PROMPT =====", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("===== END OPENAI NO-CONTEXT PROMPT =====\n", file=sys.stderr)
    raw_comment = call_openai(prompt).strip()
    return normalize_docstring_text(raw_comment)


if __name__ == "__main__":
    try:
        code_input = sys.stdin.read()
        result = generate_comment_from_code(code_input)
        print(json.dumps({"comment": result}))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
