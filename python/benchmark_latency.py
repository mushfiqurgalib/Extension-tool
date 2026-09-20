import os
import sys
import time
import json
import statistics
import csv
import urllib.request
import urllib.error

# Ensure ts/python is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from comment_generator_ollama import (
    generate_comment_from_code,
    retrieve_context,
    find_intent,
    get_ollama_base_url,
    get_ollama_model,
    ABLATION_PROFILES,
    DB_PATH,
    EMBED_MODEL
)

def benchmark_indexing():
    print("=== BENCHMARK 1: Knowledge Base Indexing Time ===")
    
    # Sample artifacts representing typical documentation / codebase files
    sample_texts = [
        "minGPT re-implementation of GPT in PyTorch. Simple, clear, and educational codebase for Transformer architectures.",
        "Module model.py: Defines GPT class, CausalSelfAttention, Block, and MLP layers.",
        "CausalSelfAttention implements multi-head masked self-attention mechanism with scale dot-product attention.",
        "Block layer combines LayerNorm, CausalSelfAttention, and MLP with residual connection pass-throughs.",
        "PyTorch Tensor operations for reshape, transpose, matrix multiplication matmul, and linear transformation layers."
    ] * 10  # 50 doc items
    
    # Measure chunking
    t0 = time.perf_counter()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = []
    for text in sample_texts:
        chunks.extend(text_splitter.split_text(text))
    t_chunking = time.perf_counter() - t0
    
    # Measure embedding creation
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks).tolist()
    t_embedding = time.perf_counter() - t0
    
    # Measure vector DB storage
    test_db_path = os.path.join(os.path.dirname(DB_PATH), "chroma_db_benchmark_temp")
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=test_db_path)
    collection = client.get_or_create_collection(name="benchmark_repo")
    
    ids = [f"bench_id_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "benchmark"} for _ in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    t_storage = time.perf_counter() - t0
    
    # Cleanup temp db
    import shutil
    try:
        shutil.rmtree(test_db_path)
    except Exception:
        pass
        
    total_indexing_time = t_chunking + t_embedding + t_storage
    avg_per_doc = total_indexing_time / len(sample_texts)
    avg_per_chunk = total_indexing_time / len(chunks)
    
    res = {
        "num_documents": len(sample_texts),
        "num_chunks": len(chunks),
        "chunking_time_sec": round(t_chunking, 4),
        "embedding_time_sec": round(t_embedding, 4),
        "db_storage_time_sec": round(t_storage, 4),
        "total_indexing_time_sec": round(total_indexing_time, 4),
        "avg_index_time_per_doc_sec": round(avg_per_doc, 4),
        "avg_index_time_per_chunk_sec": round(avg_per_chunk, 4),
    }
    print(json.dumps(res, indent=2))
    return res


def benchmark_retrieval(num_queries=20):
    print("\n=== BENCHMARK 2: Retrieval Latency (ChromaDB + Embedding) ===")
    sample_queries = [
        "how does causal self attention work in transformer",
        "docstring for neural network block layer",
        "optimizer configuration for adamw training",
        "forward pass function definition",
        "embedding matrix initialization"
    ] * (num_queries // 5)
    
    latencies = []
    for query in sample_queries:
        t0 = time.perf_counter()
        docs = retrieve_context(query, k=3)
        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed)
        
    res = {
        "num_queries": len(sample_queries),
        "mean_retrieval_latency_sec": round(statistics.mean(latencies), 4),
        "median_retrieval_latency_sec": round(statistics.median(latencies), 4),
        "min_retrieval_latency_sec": round(min(latencies), 4),
        "max_retrieval_latency_sec": round(max(latencies), 4),
        "std_retrieval_latency_sec": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
    }
    print(json.dumps(res, indent=2))
    return res


def benchmark_generation(sample_csv_path, num_samples=15):
    print(f"\n=== BENCHMARK 3: Generation Latency across Ablation Profiles (n={num_samples}) ===")
    
    if not os.path.exists(sample_csv_path):
        print(f"Sample file {sample_csv_path} not found.")
        return {}

    code_samples = []
    with open(sample_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            if code and len(code_samples) < num_samples:
                code_samples.append(code)

    profiles_to_test = [
        "phi_only",
        "phi_ast",
        "phi_ast_intent",
        "full_rag"
    ]
    
    results = {}
    
    for profile_name in profiles_to_test:
        print(f"\nTesting profile: {profile_name}...")
        times = []
        for idx, code in enumerate(code_samples, start=1):
            t0 = time.perf_counter()
            try:
                res = generate_comment_from_code(code, ablation_profile=profile_name)
                t_elapsed = time.perf_counter() - t0
                times.append(t_elapsed)
                print(f"  Sample {idx}/{len(code_samples)}: {t_elapsed:.2f}s")
            except Exception as e:
                print(f"  Sample {idx}/{len(code_samples)} failed: {e}")
                
        if times:
            results[profile_name] = {
                "num_samples": len(times),
                "mean_latency_sec": round(statistics.mean(times), 4),
                "median_latency_sec": round(statistics.median(times), 4),
                "min_latency_sec": round(min(times), 4),
                "max_latency_sec": round(max(times), 4),
                "std_latency_sec": round(statistics.stdev(times), 4) if len(times) > 1 else 0
            }
            
    print("\n--- Generation Latency Summary ---")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    csv_path = r"F:\Extension\Result\python_train_0_sample.csv"
    
    idx_res = benchmark_indexing()
    ret_res = benchmark_retrieval(num_queries=10)
    gen_res = benchmark_generation(csv_path, num_samples=10)
    
    summary = {
        "indexing_benchmark": idx_res,
        "retrieval_benchmark": ret_res,
        "generation_benchmark": gen_res
    }
    
    output_json = r"F:\Extension\Result\latency_and_indexing_benchmark_summary.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved benchmark results to {output_json}")
