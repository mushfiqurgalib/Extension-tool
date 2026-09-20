import os
import sys
import time
import json
import statistics
import csv
import urllib.request
import urllib.error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter

# Import Ollama generator
from comment_generator_ollama import (
    generate_comment_from_code as generate_comment_ollama,
    find_intent as find_intent_ollama,
    get_ollama_base_url,
    get_ollama_model,
    ABLATION_PROFILES,
    DB_PATH,
    EMBED_MODEL
)

# Import OpenAI generator
import comment_generator as openai_gen

def check_openai_key():
    openai_gen.load_env_file()
    key = os.environ.get("OPENAI_API_KEY")
    return key if key and key.strip() != "your_api_key_here" else None

def run_fast_benchmark():
    print("=" * 60)
    print("  INDEXING, OLLAMA & OPENAI GENERATION LATENCY BENCHMARK REPORT  ")
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 1. INDEXING TIME BENCHMARK (Knowledge Base Construction)
    # -------------------------------------------------------------
    print("\n[1/4] Benchmarking Knowledge Base Indexing Time...")
    
    sample_texts = [
        "minGPT re-implementation of GPT in PyTorch. Simple, clear codebase for Transformer architectures.",
        "Module model.py: Defines GPT class, CausalSelfAttention, Block, and MLP layers.",
        "CausalSelfAttention implements multi-head masked self-attention mechanism with scale dot-product attention.",
        "Block layer combines LayerNorm, CausalSelfAttention, and MLP with residual connection pass-throughs.",
        "PyTorch Tensor operations for reshape, transpose, matrix multiplication matmul, and linear transformation layers."
    ] * 10  # 50 docs
    
    t0 = time.perf_counter()
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = []
    for text in sample_texts:
        chunks.extend(splitter.split_text(text))
    t_chunking = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks).tolist()
    t_embedding = time.perf_counter() - t0
    
    test_db_path = os.path.join(os.path.dirname(DB_PATH), "chroma_db_fast_temp")
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=test_db_path)
    col = client.get_or_create_collection(name="fast_bench")
    col.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": "bench"} for _ in chunks],
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    t_storage = time.perf_counter() - t0
    
    import shutil
    try:
        shutil.rmtree(test_db_path)
    except Exception:
        pass
        
    total_idx_time = t_chunking + t_embedding + t_storage
    
    indexing_stats = {
        "num_documents": len(sample_texts),
        "num_chunks": len(chunks),
        "chunking_time_sec": round(t_chunking, 4),
        "embedding_time_sec": round(t_embedding, 4),
        "vector_db_storage_time_sec": round(t_storage, 4),
        "total_indexing_time_sec": round(total_idx_time, 4),
        "avg_index_time_per_doc_sec": round(total_idx_time / len(sample_texts), 4),
        "avg_index_time_per_chunk_sec": round(total_idx_time / len(chunks), 4),
        "throughput_chunks_per_sec": round(len(chunks) / total_idx_time, 2)
    }
    
    print("Indexing Benchmark Results:")
    print(json.dumps(indexing_stats, indent=2))
    
    # -------------------------------------------------------------
    # 2. RETRIEVAL LATENCY BENCHMARK (ChromaDB + Query Encoding)
    # -------------------------------------------------------------
    print("\n[2/4] Benchmarking RAG Retrieval Latency...")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        repo_col = chroma_client.get_collection(name="repo_context")
    except Exception:
        repo_col = None

    sample_queries = [
        "how does causal self attention work in transformer",
        "docstring for neural network block layer",
        "optimizer configuration for adamw training",
        "forward pass function definition",
        "embedding matrix initialization"
    ] * 2  # 10 queries
    
    ret_times = []
    for query in sample_queries:
        t0 = time.perf_counter()
        q_emb = model.encode([query]).tolist()
        if repo_col:
            res = repo_col.query(query_embeddings=q_emb, n_results=3)
        t_elapsed = time.perf_counter() - t0
        ret_times.append(t_elapsed)
        
    retrieval_stats = {
        "num_queries": len(sample_queries),
        "mean_retrieval_latency_sec": round(statistics.mean(ret_times), 4),
        "median_retrieval_latency_sec": round(statistics.median(ret_times), 4),
        "min_retrieval_latency_sec": round(min(ret_times), 4),
        "max_retrieval_latency_sec": round(max(ret_times), 4),
    }
    print("Retrieval Latency Results:")
    print(json.dumps(retrieval_stats, indent=2))

    # Load code samples for generation benchmarks
    sample_csv = r"F:\Extension\Result\python_train_0_sample.csv"
    code_samples = []
    if os.path.exists(sample_csv):
        with open(sample_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip()
                if code and len(code_samples) < 3:
                    code_samples.append(code)

    # -------------------------------------------------------------
    # 3. GENERATION LATENCY BENCHMARK (Ollama)
    # -------------------------------------------------------------
    print("\n[3/4] Benchmarking Ollama Generation Latency...")
    profiles = ["phi_only", "phi_ast", "phi_ast_intent", "full_rag"]
    ollama_profile_results = {}
    
    for prof in profiles:
        prof_times = []
        for code in code_samples:
            t0 = time.perf_counter()
            _ = generate_comment_ollama(code, ablation_profile=prof)
            prof_times.append(time.perf_counter() - t0)
            
        ollama_profile_results[prof] = {
            "num_samples": len(prof_times),
            "mean_generation_latency_sec": round(statistics.mean(prof_times), 4),
            "median_generation_latency_sec": round(statistics.median(prof_times), 4),
            "min_generation_latency_sec": round(min(prof_times), 4),
            "max_generation_latency_sec": round(max(prof_times), 4),
        }

    ollama_summary = {
        "model": get_ollama_model(),
        "profiles": ollama_profile_results
    }
    print("Ollama Generation Latency:")
    print(json.dumps(ollama_summary, indent=2))

    # -------------------------------------------------------------
    # 4. GENERATION LATENCY BENCHMARK (OpenAI)
    # -------------------------------------------------------------
    print("\n[4/4] Benchmarking OpenAI Generation Latency...")
    openai_key = check_openai_key()
    openai_summary = {}
    
    if openai_key:
        print(f"OpenAI API key detected. Testing OpenAI model: {os.environ.get('OPENAI_MODEL', 'gpt-5.4-mini')}...")
        openai_times = []
        for code in code_samples:
            t0 = time.perf_counter()
            try:
                _ = openai_gen.generate_comment_from_code(code)
                openai_times.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"OpenAI generation call failed: {e}")
                
        if openai_times:
            openai_summary = {
                "status": "success",
                "model": os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
                "num_samples": len(openai_times),
                "mean_generation_latency_sec": round(statistics.mean(openai_times), 4),
                "median_generation_latency_sec": round(statistics.median(openai_times), 4),
                "min_generation_latency_sec": round(min(openai_times), 4),
                "max_generation_latency_sec": round(max(openai_times), 4),
            }
        else:
            openai_summary = {
                "status": "failed",
                "error": "OpenAI API calls returned errors during benchmark execution."
            }
    else:
        print("OPENAI_API_KEY is not configured in environment or ts/.env.")
        openai_summary = {
            "status": "missing_api_key",
            "message": "OPENAI_API_KEY environment variable is not set. Set OPENAI_API_KEY in ts/.env to benchmark OpenAI API latency."
        }
        
    print("OpenAI Generation Latency Results:")
    print(json.dumps(openai_summary, indent=2))

    full_report = {
        "indexing_benchmark": indexing_stats,
        "retrieval_benchmark": retrieval_stats,
        "generation_benchmark_ollama": ollama_summary,
        "generation_benchmark_openai": openai_summary
    }
    
    out_file = r"F:\Extension\Result\latency_and_indexing_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\nSaved benchmark report to: {out_file}")
    return full_report

if __name__ == "__main__":
    run_fast_benchmark()
