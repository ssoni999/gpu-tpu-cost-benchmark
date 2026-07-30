#!/usr/bin/env python3
import asyncio
import argparse
import time
import os
import json
import statistics
import aiohttp

# =====================================================================
# HARDWARE & INFRASTRUCTURE AUTO-DETECTION
# =====================================================================

PEAK_TFLOPS_DEFAULTS = {
    "TPU v5e": 197.0,  # Peak BF16/INT8 TFLOPS per TPU v5e chip
    "GPU (T4)": 70.0,   # Peak FP16 TFLOPS for Nvidia T4
}

def detect_infrastructure():
    """Detects hardware environment based on system devices."""
    if os.path.exists("/dev/accel0") or "TPU_NAME" in os.environ:
        return "TPU v5e"
    elif os.path.exists("/dev/nvidia0") or "CUDA_VISIBLE_DEVICES" in os.environ:
        return "GPU (T4)"
    return "TPU v5e"

# =====================================================================
# SERVER-SIDE PROMETHEUS METRIC SCRAPER
# =====================================================================

async def scrape_cumulative_metrics(session, url, retries=3):
    """
    Scrapes cumulative metrics from the vLLM Prometheus endpoint.
    Includes retry logic and timeout adjustments to handle port-forward drops.
    """
    server_base = url.rsplit('/v1', 1)[0] if '/v1' in url else url.rsplit('/', 1)[0]
    metrics_url = f"{server_base.rstrip('/')}/metrics"
    
    for attempt in range(retries):
        try:
            async with session.get(metrics_url, timeout=5.0) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    
                    kv_usage = 0.0
                    queued_reqs = 0.0
                    running_reqs = 0.0
                    queries = 0.0
                    hits = 0.0

                    for line in content.splitlines():
                        if line.startswith("#") or not line.strip():
                            continue
                        
                        # KV Cache Usage
                        if "vllm:kv_cache_usage_perc" in line or "vllm:gpu_cache_usage_perc" in line:
                            val = float(line.split()[-1])
                            kv_usage = val * 100.0 if val <= 1.0 else val
                        
                        # Queued and Running Requests
                        elif "vllm:num_requests_waiting" in line:
                            queued_reqs = float(line.split()[-1])
                        elif "vllm:num_requests_running" in line:
                            running_reqs = float(line.split()[-1])
                        
                        # Prefix Cache Metrics
                        elif "vllm:prefix_cache_queries_total" in line:
                            queries = float(line.split()[-1])
                        elif "vllm:prefix_cache_hits_total" in line:
                            hits = float(line.split()[-1])

                    hit_rate = (hits / queries * 100.0) if queries > 0 else 0.0
                    return {
                        "status": "success",
                        "kv_usage_pct": kv_usage,
                        "queued_requests": queued_reqs,
                        "running_requests": running_reqs,
                        "prefix_hit_rate_pct": hit_rate,
                        "prefix_queries": queries,
                        "prefix_hits": hits
                    }
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(0.5)
            else:
                return {
                    "status": f"unreachable ({type(e).__name__})",
                    "kv_usage_pct": 0.0,
                    "queued_requests": 0.0,
                    "running_requests": 0.0,
                    "prefix_hit_rate_pct": 0.0,
                    "prefix_queries": 0.0,
                    "prefix_hits": 0.0
                }

async def monitor_metrics(session, url, metrics_log, stop_event, interval=0.2):
    """Background task to continuously poll server metrics during workload execution."""
    while not stop_event.is_set():
        data = await scrape_cumulative_metrics(session, url)
        if data["status"] == "success":
            metrics_log.append(data)
        await asyncio.sleep(interval)

# =====================================================================
# REQUEST WORKER
# =====================================================================

async def send_request(session, url, model, system_prompt, user_prompt, max_tokens, request_id, total_requests, infra_type, temperature, top_p, top_k):
    """Sends a streaming request with system + user prompts and records latency metrics."""
    headers = {"Content-Type": "application/json"}
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = start_time
    token_count = 0
    token_times = []

    try:
        async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
            if resp.status != 200:
                return {"success": False, "error": f"HTTP {resp.status}"}

            async for line in resp.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        token_times.append(now - last_token_time)
                    
                    last_token_time = now
                    token_count += 1

            end_time = time.perf_counter()
            e2e_latency = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else e2e_latency
            avg_itl = statistics.mean(token_times) if token_times else 0.0

            log_interval = max(1, total_requests // 5)
            if (request_id + 1) % log_interval == 0 or request_id == 0 or (request_id + 1) == total_requests:
                print(f"[{infra_type} | Trace {request_id+1}/{total_requests}] TTFT: {round(ttft*1000, 1)}ms | E2E Latency: {round(e2e_latency, 3)}s")

            return {
                "success": True,
                "ttft": ttft,
                "e2e_latency": e2e_latency,
                "token_count": token_count,
                "avg_itl": avg_itl,
                "tps": (token_count / e2e_latency) if e2e_latency > 0 else 0.0
            }

    except Exception as e:
        print(f"[{infra_type} | Trace {request_id+1}] ERROR: {e}")
        return {"success": False, "error": str(e)}

# =====================================================================
# BENCHMARK RUNNER
# =====================================================================

async def run_benchmark(args):
    url = args.url
    model = args.model
    concurrency = args.concurrency
    max_tokens = args.max_tokens
    params_b = args.params_b
    num_requests = args.num_requests
    system_prompt = args.system_prompt
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k

    infra_type = detect_infrastructure()
    peak_flops = PEAK_TFLOPS_DEFAULTS.get(infra_type, 100.0)

    # Base categorized prompt templates
    categories = {
        'Coding': [
            'Write a binary search in Python.',
            'Explain recursion with code.',
            'Fix a memory leak in C++.',
            'Implement a LRU cache in Python.',
            'Write a function to invert a binary tree.',
            'Implement quicksort in Go.',
            'Write a function to check if a string is a palindrome.',
            'Implement a thread-safe singleton in Java.',
            'Write an async web scraper in Python.',
            'Implement a Trie (prefix tree) data structure.'
        ],
        'Science': [
            'How does photosynthesis work?',
            'What is quantum entanglement?',
            'How do solar panels work?',
            'Explain the second law of thermodynamics.',
            'What causes gravity according to general relativity?',
            'How do vaccines trigger an immune response?',
            'Explain CRISPR gene editing in simple terms.',
            'What is the difference between nuclear fission and fusion?',
            'How do lithium-ion batteries store energy?',
            'Explain the Doppler effect with a real-world example.'
        ],
        'DevOps': [
            'Docker vs VMs summary.',
            'Explain Kubernetes ingress rules.',
            'What is GitOps?',
            'How does blue-green deployment work?',
            'Explain canary releases vs rolling deployments.',
            'What is Terraform state and why is it needed?',
            'How does Prometheus collect metrics from targets?',
            'Explain the role of a reverse proxy like Nginx.',
            'What is distributed tracing in observability?',
            'How do Cgroups and Namespaces isolate container processes?'
        ],
        'Math': [
            'Derivative of x^2 * sin(x)?',
            'Explain the Fibonacci sequence.',
            'How do prime numbers work?',
            'Explain Bayes Theorem with an intuitive example.',
            'What is Eigenvalue decomposition used for?',
            'Explain the central limit theorem.',
            'How does gradient descent minimize loss functions?',
            'What is the physical meaning of a dot product vs cross product?',
            'Calculate the integral of e^(-x) from 0 to infinity.',
            'Explain the concept of P vs NP in computational theory.'
        ],
        'SQL': [
            'Query for 2nd highest salary.',
            'INNER vs LEFT JOIN.',
            'Explain indexing in Postgres.',
            'Write a CTE query to calculate rolling averages.',
            'Explain window functions like ROW_NUMBER() and RANK().',
            'How do ACID properties guarantee database transactions?',
            'Explain database sharding vs read replicas.',
            'How do B-Tree indexes speed up lookups?',
            'Query to find duplicate records in a table.',
            'Explain optimistic vs pessimistic locking.'
        ],
        'Summarization': [
            'Summarize Hamlet in 3 sentences.',
            'Summarize how the internet works.',
            'Summarize machine learning.',
            'Summarize the CAP theorem in database design.',
            'Summarize the history of Cloud Computing in 4 bullets.',
            'Summarize how the TCP/IP 3-way handshake works.',
            'Summarize the core concept of Raft consensus algorithm.',
            'Summarize the difference between synchronous and asynchronous execution.',
            'Summarize the impact of Moore Law on modern computing.',
            'Summarize how Transformer attention mechanisms process sequence inputs.'
        ],
        'Translation': [
            'Translate "Hello, how are you?" to French, Spanish, German.',
            'Translate greeting to Japanese.',
            'Translate "Cloud infrastructure optimization" to Mandarin.',
            'Translate "Error 500 Internal Server Failure" to Italian.',
            'Translate "High availability and zero downtime" to Portuguese.',
            'Translate "Automated pipeline deployed successfully" to Dutch.',
            'Translate "Data encryption at rest and in transit" to Korean.',
            'Translate "Latency reduced by 40 percent" to Russian.',
            'Translate "Distributed system architecture" to Arabic.',
            'Translate "System performance monitoring" to Hindi.'
        ],
        'Creative': [
            'Write a poem about Cloud TPUs.',
            '2-sentence story about space.',
            'Write a haiku about rain.',
            'Write a witty dialogue between a GPU and a CPU.',
            'Describe a futuristic AI city in 3 descriptive sentences.',
            'Write a sci-fi micro-fiction about quantum teleporters.',
            'Write a short fable about a slow algorithm learning to scale.',
            'Compose a limerick about Kubernetes pods.',
            'Write a metaphorical description of compiler optimizations.',
            'Draft a fictional release note for teleportation firmware v1.0.'
        ],
        'Business': [
            '3 bullet summary on customer retention.',
            'Top metrics for SaaS startups.',
            'Explain CAC vs LTV in unit economics.',
            'What is product-led growth (PLG)?',
            'How do Service Level Agreements (SLAs) differ from SLOs and SLIs?',
            'Explain churn rate and how to minimize it.',
            'What is gross margin and why is it critical for tech firms?',
            'Explain the concept of a Minimum Viable Product (MVP).',
            'What is net revenue retention (NRR)?',
            'Explain competitive advantage through Network Effects.'
        ],
        'Trivia': [
            'What are the 7 continents?',
            'Element with atomic number 1?',
            'Who painted the Mona Lisa?',
            'What is the capital of Australia?',
            'Who invented the World Wide Web?',
            'Which planet has the most moons in our solar system?',
            'What year was Python released?',
            'What is the fastest land animal?',
            'Who authored the Linux kernel?',
            'What is the deepest known location in the oceans?'
        ],
        'System Design': [
            'Design a rate limiter for a REST API.',
            'How would you design a URL shortener like bit.ly?',
            'Design a distributed message queue like Kafka.',
            'How to design a scalable notification service?',
            'Design a real-time collaborative document editor like Google Docs.',
            'How to design a global CDN caching architecture?',
            'Design a metric telemetry monitoring system.',
            'How to handle consensus in distributed databases?',
            'Design a WebSockets cluster for real-time chat.',
            'How to mitigate single points of failure in cloud architecture?'
        ],
        'Security': [
            'How does TLS/SSL handshake establish secure connections?',
            'Explain OAuth 2.0 vs OpenID Connect.',
            'What is SQL injection and how do you prevent it?',
            'Explain Cross-Site Scripting (XSS) and mitigation strategies.',
            'How does public-key cryptography work?',
            'What is zero trust security model?',
            'Explain JWT structure and signature verification.',
            'How do buffer overflows compromise system memory?',
            'What is secret management in Kubernetes (HashiCorp Vault vs K8s Secrets)?',
            'How does DDOS mitigation work at L4 vs L7 OSI layers?'
        ],
        'Data Engineering': [
            'Batch vs Stream processing (Spark vs Flink).',
            'What is an ETL pipeline and how to handle failures gracefully?',
            'Explain Star Schema vs Snowflake Schema in data warehousing.',
            'How does Apache Parquet column-oriented storage save space?',
            'What is data lineage and why is it important?',
            'Explain data lakehouse architecture vs traditional data warehouses.',
            'How does dead letter queue (DLQ) handle corrupted data pipelines?',
            'What is partitioning vs clustering in BigQuery/Snowflake?',
            'Explain backpressure handling in real-time data streams.',
            'How does CDC (Change Data Capture) work for database replication?'
        ],
        'ML/AI Systems': [
            'Difference between FP32, FP16, and INT8 quantization.',
            'Explain Key-Value (KV) cache in transformer inference engines.',
            'What is model parallelism vs data parallelism in distributed training?',
            'Explain LoRA (Low-Rank Adaptation) fine-tuning.',
            'What is speculative decoding in LLM serving?',
            'How does FlashAttention optimize GPU memory bandwidth?',
            'Explain continuous batching vs static batching in vLLM.',
            'What causes GPU memory fragmentation during LLM inference?',
            'Explain Retrieval-Augmented Generation (RAG) architecture.',
            'How does mixture of experts (MoE) routing work?'
        ],
        'API Design': [
            'REST vs GraphQL vs gRPC comparison.',
            'How to handle API versioning cleanly?',
            'Best practices for HTTP error status codes (4xx vs 5xx).',
            'How to design idempotent POST/PUT endpoints using idempotency keys?',
            'Explain Webhooks architecture for event-driven integrations.',
            'How to implement cursor-based pagination vs offset pagination?',
            'What is OpenAPI / Swagger specification?',
            'How to secure public APIs against scraping and abuse?',
            'Explain backward-compatible schema changes in Protobuf.',
            'How to structure health checks for Kubernetes readiness endpoints?'
        ]
    }

    raw_prompts = []
    cat_keys = list(categories.keys())
    for i in range(num_requests):
        cat_name = cat_keys[i % len(cat_keys)]
        prompt_template = categories[cat_name][i % len(categories[cat_name])]
        raw_prompts.append(f"[{cat_name}] {prompt_template} (Trace #{i+1})")

    print("=" * 60)
    print(f" STARTING BENCHMARK: {num_requests} Requests | Concurrency: {concurrency}")
    print(f" Infrastructure   : {infra_type} (Peak Capacity: {peak_flops} TFLOPS)")
    print(f" Target Endpoint  : {url}")
    print(f" Model           : {model} ({params_b}B Params)")
    print(f" Temperature     : {temperature}")
    print(f" Top_k           : {top_k}")
    print(f" Top_p           : {top_p}")
    print(f" System Prompt   : {'Enabled (' + str(len(system_prompt)) + ' chars)' if system_prompt else 'None'}")
    print("=" * 60 + "\n")

    semaphore = asyncio.Semaphore(concurrency)
    metrics_log = []
    stop_event = asyncio.Event()

    async with aiohttp.ClientSession() as session:
        monitor_task = asyncio.create_task(
            monitor_metrics(session, url, metrics_log, stop_event, interval=0.2)
        )

        async def worker(req_id, user_prompt_text):
            async with semaphore:
                return await send_request(
                    session, url, model, system_prompt, user_prompt_text, 
                    max_tokens, req_id, num_requests, infra_type,
                    temperature, top_p, top_k
                )

        wall_start = time.perf_counter()
        tasks = [worker(i, prompt) for i, prompt in enumerate(raw_prompts)]
        results = await asyncio.gather(*tasks)
        wall_duration = time.perf_counter() - wall_start

        stop_event.set()
        await monitor_task

        final_metric = await scrape_cumulative_metrics(session, url)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    ttfts = [r["ttft"] for r in successful]
    e2e_latencies = [r["e2e_latency"] for r in successful]
    total_tokens = sum(r["token_count"] for r in successful)
    itls = [r["avg_itl"] for r in successful if r["avg_itl"] > 0]

    ttft_p50 = statistics.median(ttfts) if ttfts else 0.0
    ttft_p95 = statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else ttft_p50
    
    e2e_p50 = statistics.median(e2e_latencies) if e2e_latencies else 0.0
    e2e_p95 = statistics.quantiles(e2e_latencies, n=20)[18] if len(e2e_latencies) >= 20 else e2e_p50
    avg_e2e = statistics.mean(e2e_latencies) if e2e_latencies else 0.0

    avg_itl = statistics.mean(itls) if itls else 0.0
    overall_tps = total_tokens / wall_duration if wall_duration > 0 else 0.0
    overall_rps = num_requests / wall_duration if wall_duration > 0 else 0.0

    achieved_tflops = (overall_tps * 2.0 * params_b * 1e9) / 1e12
    mfu_percent = (achieved_tflops / peak_flops * 100.0) if peak_flops > 0 else 0.0

    peak_kv = max([m["kv_usage_pct"] for m in metrics_log], default=final_metric["kv_usage_pct"])
    peak_queued = max([m["queued_requests"] for m in metrics_log], default=final_metric["queued_requests"])
    peak_active = max([m["running_requests"] for m in metrics_log], default=final_metric["running_requests"])
    prefix_hit_rate = final_metric["prefix_hit_rate_pct"]

    # Format output text block
    summary_text = f"""
============================================================
 WORKLOAD BENCHMARK SUMMARY
============================================================
 Infrastructure Environment : {infra_type}
 Total Wall-Clock Time      : {wall_duration:.2f} s
 Successful Requests        : {len(successful)} / {num_requests}
 Failed Requests            : {len(failed)}
 Throughput (RPS)           : {overall_rps:.2f} req/s
 Generated Tokens           : {total_tokens} tokens
 Throughput (TPS)           : {overall_tps:.2f} tokens/s
------------------------------------------------------------
 HARDWARE UTILIZATION METRICS:
   Model Parameter Count    : {params_b:.2f} Billion
   Achieved TFLOPS          : {achieved_tflops:.4f} TFLOPS
   Model FLOPs Util (MFU)   : {mfu_percent:.2f} % (vs {peak_flops} Peak TFLOPS)
------------------------------------------------------------
 CLIENT-SIDE LATENCY METRICS:
   TTFT P50 (Median)        : {ttft_p50 * 1000:.2f} ms
   TTFT P95                 : {ttft_p95 * 1000:.2f} ms
   Avg ITL (per-token)      : {avg_itl * 1000:.2f} ms
   E2E Latency Avg          : {avg_e2e:.2f} s
   E2E Latency P50 (Median) : {e2e_p50:.2f} s
   E2E Latency P95          : {e2e_p95:.2f} s
------------------------------------------------------------
 SERVER-SIDE ENGINE METRICS (/metrics):
   Metrics Endpoint Status  : {final_metric['status']}
   Peak KV Cache Usage      : {peak_kv:.2f} %
   Prefix Cache Hit Rate    : {prefix_hit_rate:.1f} % ({int(final_metric['prefix_hits'])} / {int(final_metric['prefix_queries'])} tokens)
   Peak Queued Requests     : {peak_queued:.1f}
   Peak Active Requests     : {peak_active:.1f}
============================================================
"""
    print(summary_text)

    # Prepare export dictionary
    export_data = {
        "config": {
            "url": url,
            "model": model,
            "concurrency": concurrency,
            "num_requests": num_requests,
            "max_tokens": max_tokens,
            "params_b": params_b,
            "infra_type": infra_type,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        },
        "summary": {
            "total_wall_clock_sec": wall_duration,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "throughput_rps": overall_rps,
            "total_generated_tokens": total_tokens,
            "throughput_tps": overall_tps,
            "achieved_tflops": achieved_tflops,
            "mfu_percent": mfu_percent,
            "ttft_p50_ms": ttft_p50 * 1000,
            "ttft_p95_ms": ttft_p95 * 1000,
            "avg_itl_ms": avg_itl * 1000,
            "e2e_latency_avg_sec": avg_e2e,
            "e2e_latency_p50_sec": e2e_p50,
            "e2e_latency_p95_sec": e2e_p95,
            "peak_kv_cache_pct": peak_kv,
            "prefix_cache_hit_rate_pct": prefix_hit_rate,
            "peak_queued_requests": peak_queued,
            "peak_active_requests": peak_active
        }
    }

    # =====================================================================
    # FILE EXPORT & DYNAMIC FILENAMING
    # =====================================================================

    results_dir = "workload_run_metrics"
    os.makedirs(results_dir, exist_ok=True)
    # Sanitize infrastructure name for filename safety (e.g., "TPU v5e" -> "TPUv5e")
    safe_infra = infra_type.replace(" ", "").replace("(", "").replace(")", "")
    
    # Order: infra, concurrency, temperature, top_p, top_k, num_requests
    auto_filename_base = f"benchmark_{safe_infra}_c{concurrency}_temp{temperature}_p{top_p}_k{top_k}_req{num_requests}"

    # Determine TXT file path
    if args.output_txt:
        txt_path = args.output_txt
    elif args.save_files:
        txt_path = os.path.join(results_dir, f"{auto_filename_base}.txt")
    else:
        txt_path = None

    # Determine JSON file path
    if args.output_json:
        json_path = args.output_json
    elif args.save_files:
        json_path = os.path.join(results_dir, f"{auto_filename_base}.json")
    else:
        json_path = None

    # Save outputs if enabled
    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"[Export] Saved text summary to: {txt_path}")

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"[Export] Saved JSON metrics to: {json_path}")

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI assistant optimized for speed, precision, and accuracy. "
    "Follow these guidelines strictly:\n"
    "1. Provide direct, highly structured, and concise answers without unnecessary filler, greetings, or conversational preamble.\n"
    "2. For coding and technical tasks, prioritize readable, performant code blocks with brief explanatory notes.\n"
    "3. For reasoning, math, and factual queries, explain core logic step-by-step using clear formatting.\n"
    "4. Keep all responses under the requested output token constraints while maintaining complete, high-quality information."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async vLLM Workload Benchmark Tool")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/chat/completions", help="Target API endpoint")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--params-b", type=float, default=0.49, help="Model parameters in billions (for TFLOPS calculation)")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens per request")
    parser.add_argument("--num-requests", type=int, default=100, help="Total number of request traces to generate and run")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt used across all requests to enable prefix caching")

    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling cutoff")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")

    # Output file export options
    parser.add_argument("--save-files", action="store_true", default=True, help="Auto-generate and save .json and .txt files with metadata filenames")
    parser.add_argument("--output-json", type=str, default=None, help="Explicit file path to save metrics as JSON")
    parser.add_argument("--output-txt", type=str, default=None, help="Explicit file path to save console summary as TXT")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))