import ast
import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = "f:/Thesis_methodology"
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_FILE = os.path.join(BASE_DIR, "minGPT", "mingpt", "model.py")

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
    In a production RAG pipeline, this would call a model like GPT-4 or CodeT5
     to summarize the logical intent of the code block.
    """
    # Simulated Phase 3 output
    return (
        "Intent: Implement multi-head causal self-attention. "
        "Logic: Linearly project input to Q, K, V; split into heads; "
        "compute scaled dot-product attention with causal triangular masking; "
        "apply dropout; concatenate heads and project back."
    )


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
    return prompt



def generate_comment_from_code(code_text):
    # Simplified: reuse your pipeline
    intent = find_intent(code_text)
    ast_tree = ast.parse(code_text)
    ast_str = flatten_ast(ast_tree)

    try:
        context = retrieve_context(code_text)
    except:
        context = ["No context found"]

    prompt = construct_unified_prompt(code_text, ast_str, context, intent)

    # 👉 HERE you would call LLM (replace with real API)
    generated_comment = "'''Generated docstring based on code'''"

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
    code_input = sys.stdin.read()
    result = generate_comment_from_code(code_input)

    print(json.dumps({"comment": result}))