import re
import threading
import time
import csv
from datetime import datetime
from fastapi import FastAPI, Request
import httpx
import uvicorn
from threading import Lock
from contextlib import asynccontextmanager

MODEL_ROUTES = {
    "casperhansen/llama-3-8b-instruct-awq": ["http://localhost:8005"],
    #  "Qwen/Qwen2.5-3B-Instruct": ["http://localhost:8007"],    
    #  "meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8006"],
    # "Qwen/Qwen2.5-14B-Instruct-AWQ": ["http://localhost:8011"],
    # "Qwen/Qwen2.5-7B-Instruct": ["http://localhost:8005"],
    # "Qwen/Qwen2.5-0.5B-Instruct": ["http://localhost:8011"],
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ["http://localhost:8006"],
    # "Qwen/Qwen2.5-7B-Instruct": ["http://localhost:8006"],
    
}


BACKEND_LOGS = {
    "http://localhost:8005": "/scratch/global/palimpzest/abacus-research/var/logs/saida_VLLM_8005.txt",
}

BACKEND_METRICS = {
    url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
    for url in BACKEND_LOGS
}

LOG_PATTERN = re.compile(
    r"Running:\s*(\d+).*?Waiting:\s*(\d+).*?GPU KV cache usage:\s*([0-9.]+)"
)

# LOG_PATTERN = re.compile(
#     r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%"
# )

metrics_lock = Lock()

# --- Métricas globais ---
LOG_FILE = "./proxy_metrics.csv"
log_lock = Lock()
monitoring_active = False
first_request_time = None


# def log_latency(latency, backend_url):
#     """Salva a latência da requisição em CSV"""
#     with log_lock:
#         with open(LOG_FILE, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([datetime.now().isoformat(), latency, backend_url])


def tail_log_file(path, pattern):
    """Yield new log lines matching pattern."""
    try:
        with open(path, "r") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if pattern.search(line):
                    yield line
    except FileNotFoundError:
        print(f"[WARN] Log file {path} not found")
        while True:
            time.sleep(5)

def monitor_logs(backend_url, log_path):
    """Background thread robusta para monitorar logs."""
    print(f"[INFO] Monitoring started for {backend_url} at {log_path}")
    for line in tail_log_file(log_path, LOG_PATTERN):
        # Remove espaços extras e caracteres estranhos
        clean_line = line.strip()
        match = LOG_PATTERN.search(clean_line)
        if match:
            running, waiting, kv_cache = match.groups()
            with metrics_lock:
                BACKEND_METRICS[backend_url]["running"] = int(running)
                BACKEND_METRICS[backend_url]["waiting"] = int(waiting)
                BACKEND_METRICS[backend_url]["kv_cache"] = float(kv_cache)
                BACKEND_METRICS[backend_url]["last_updated"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start monitoring threads when app starts."""
    print("[INFO] Starting log monitoring threads...")
    for backend_url, path in BACKEND_LOGS.items():
        t = threading.Thread(target=monitor_logs, args=(backend_url, path), daemon=True)
        t.start()
    yield
    print("[INFO] Shutting down proxy...")


app = FastAPI(lifespan=lifespan)


def print_backend_status():
    """Print formatted metrics for all backends."""
    with metrics_lock:
        print("\n[STATUS] Current backend load:")
        print("-" * 70)
        for url, metrics in BACKEND_METRICS.items():
            port = url.split(":")[-1]
            running = metrics["running"]
            waiting = metrics["waiting"]
            kv = metrics["kv_cache"]
            updated_ago = time.time() - metrics["last_updated"]
            print(
                f"  • Port {port}: running={running:2d}, waiting={waiting:2d}, "
                f"KV={kv:5.1f}%, updated {updated_ago:4.1f}s ago"
            )
        print("-" * 70 + "\n")


@app.get("/status")
async def get_status():
    """Return current backend metrics for visualization."""
    with metrics_lock:
        return BACKEND_METRICS


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    global monitoring_active, first_request_time

    start_time = time.time()
    body = await request.json()
    model = body.get("model")

    temperature = body.get("temperature", 0.0)
    # Ativa o monitoramento somente quando chega a primeira requisição
    if not monitoring_active:
        monitoring_active = True
        first_request_time = start_time
        print(
            f"[METRICS] Monitoring started at {datetime.now().isoformat()} "
            "(first request received)"
        )

    if model not in MODEL_ROUTES:
        return {"error": f"Unknown model: {model}"}

    candidates = MODEL_ROUTES[model]

    with metrics_lock:
        current_metrics = {url: BACKEND_METRICS[url].copy() for url in candidates}

    print(f"\n[REQUEST] Received new request for model: {model} | Temperature: {temperature}")
    print_backend_status()

    now = time.time()
    valid_backends = [
        url for url in candidates
        if now - current_metrics[url].get("last_updated", 0) < 120
    ]

    if not valid_backends:
        best_backend = candidates[0]
        print(f"[WARN] No valid metrics for {model}, fallback to {best_backend}")
    else:
        best_backend = min(
            valid_backends,
            key=lambda url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
        )

    total_loads = {
        url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
        for url in valid_backends
    }

    print(f"[INFO] Load summary: {total_loads}")
    print(f"[INFO] Selected backend for {model}: {best_backend}\n")

    target_url = f"{best_backend}/v1/chat/completions"
    
    try:
        async with httpx.AsyncClient(timeout=36000.0) as client:
            resp = await client.post(target_url, json=body)

        latency = time.time() - start_time
        # log_latency(latency, best_backend)  # salva no CSV
        print(f"[METRIC] E2E latency = {latency:.3f}s | Backend = {best_backend}")

        return resp.json()
    except httpx.RequestError as e:
        latency = time.time() - start_time
        # log_latency(latency, best_backend)
        print(f"[ERROR] Failed request ({latency:.3f}s): {e}")
        return {"error": f"Backend unavailable: {best_backend}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
