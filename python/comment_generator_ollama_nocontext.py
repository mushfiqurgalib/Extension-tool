import json
import os
import sys
import urllib.error
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


def get_ollama_base_url():
    load_env_file()
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ollama_model():
    load_env_file()
    return os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")


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
        headers={"Content-Type": "application/json"},
        method="POST",
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


def build_prompt(code_text):
    return f"""
### RAW LEXICAL CODE
```python
{code_text}
```

### TASK
Generate a highly accurate, professional NumPy-style docstring comment for the function above,
reflecting the specific mathematical intent and framework versions found in the context.

### OUTPUT
Return only the final Python docstring.
Do not include explanations, markdown fences, or any surrounding text.
"""


def generate_comment_from_code(code_text):
    prompt = build_prompt(code_text)
    print("\n===== OLLAMA NO-CONTEXT PROMPT =====", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("===== END OLLAMA NO-CONTEXT PROMPT =====\n", file=sys.stderr)
    return call_ollama(prompt).strip()


if __name__ == "__main__":
    try:
        code_input = sys.stdin.read()
        result = generate_comment_from_code(code_input)
        print(json.dumps({"comment": result}))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
