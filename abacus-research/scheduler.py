# import re
# import threading
# import time
# import csv
# from datetime import datetime
# from fastapi import FastAPI, Request
# import httpx
# import uvicorn
# from threading import Lock
# from contextlib import asynccontextmanager

# MODEL_ROUTES = {
#     # "meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8005"],
#     # "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8005"],
#     # "Qwen/Qwen3-4B": ["http://localhost:8007"],
#     # "Qwen/Qwen2.5-14B-Instruct-AWQ": ["http://localhost:8005"],
#     # "Qwen/Qwen2.5-7B-Instruct": ["http://localhost:8005"],
#     # "hosted_vllm/Qwen/Qwen2.5-7B-Instruct": ["http://localhost:8005"],
#     # "Qwen/Qwen2.5-1.5B-Instruct": ["http://localhost:8005"],
#     # "hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct": ["http://localhost:8005"],
#     # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ["http://localhost:8006"],
#     # "Qwen/Qwen2.5-7B-Instruct": ["http://localhost:8006"],
#     # "Qwen/Qwen3-0.6B": ["http://localhost:8006"],
#     # "Qwen/Qwen2.5-0.5B-Instruct": ["http://localhost:8007"],
#     # "Qwen/Qwen2.5-14B-Instruct-AWQ": ["http://localhost:8006"],

#     # NAMES_AXIA
#     # "meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8005"],
#     "meta-llama/Llama-3.3-70B-Instruct": ["http://localhost:8006"],
#     # "Qwen/Qwen3-235B-A22B": ["http://localhost:8007"],
#     # "openai/gpt-oss-120b": ["http://localhost:8008"],
# }


  
# BACKEND_LOGS = {
#     # "http://localhost:8005": "./var/logs/saida_VLLM_8005.txt",
#     "http://localhost:8006": "./var/logs/saida_VLLM_8006.txt",
#     # "http://localhost:8007": "./var/logs/saida_VLLM_8007.txt",
#     # "http://localhost:8008": "./var/logs/saida_VLLM_8008.txt",
# }

# # BACKEND_METRICS = {
# #     url: {"running": 0, "waiting": 0, "kv_cache": 0.0, "last_updated": 0}
# #     for url in BACKEND_LOGS
# # }

# # LOG_PATTERN = re.compile(
# #     r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%"
# # )

# # ------------------------------- Abaixo script alterado ------------------------------------
# BACKEND_METRICS = {
#     url: {
#         "running": 0, 
#         "waiting": 0, 
#         "kv_cache": 0.0, 
#         "last_updated": time.time()  # <--- MUDE DE 0 PARA time.time()
#     }
#     for url in BACKEND_LOGS
# }

# LOG_PATTERN = re.compile(
#     r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs,.*?(?:GPU KV cache usage:|KV cache usage:)\s*([0-9.]+|N/A)%"
# )
# #------------------------------------ Fim da atualização ------------------------------------

# metrics_lock = Lock()

# # --- Métricas globais ---
# LOG_FILE = "./proxy_metrics.csv"
# log_lock = Lock()
# monitoring_active = False
# first_request_time = None


# # def log_latency(latency, backend_url):
# #     """Salva a latência da requisição em CSV"""
# #     with log_lock:
# #         with open(LOG_FILE, "a", newline="") as f:
# #             writer = csv.writer(f)
# #             writer.writerow([datetime.now().isoformat(), latency, backend_url])


# def tail_log_file(path, pattern):
#     """Yield new log lines matching pattern."""
#     try:
#         with open(path, "r") as f:
#             f.seek(0, 2)
#             while True:
#                 line = f.readline()
#                 if not line:
#                     time.sleep(0.5)
#                     continue
#                 if pattern.search(line):
#                     yield line
#     except FileNotFoundError:
#         print(f"[WARN] Log file {path} not found")
#         while True:
#             time.sleep(5)


# # def monitor_logs(backend_url, log_path):
# #     """Background thread that updates metrics from log."""
# #     for line in tail_log_file(log_path, LOG_PATTERN):
# #         match = LOG_PATTERN.search(line)
# #         if match:
# #             running, waiting, kv_cache = match.groups()
# #             with metrics_lock:
# #                 BACKEND_METRICS[backend_url]["running"] = int(running)
# #                 BACKEND_METRICS[backend_url]["waiting"] = int(waiting)
# #                 BACKEND_METRICS[backend_url]["kv_cache"] = float(kv_cache)
# #                 BACKEND_METRICS[backend_url]["last_updated"] = time.time()

# # ------------------- Script atualizado para lidar com "N/A" no KV cache -------------------
# def monitor_logs(backend_url, log_path):
#     """Thread que monitora os logs e atualiza as métricas."""
#     print(f"[DEBUG] Iniciando tail no log: {log_path} para o backend: {backend_url}")
#     for line in tail_log_file(log_path, LOG_PATTERN):
#         match = LOG_PATTERN.search(line)
#         if match:
#             running, waiting, kv_cache = match.groups()
#             with metrics_lock:
#                 BACKEND_METRICS[backend_url]["running"] = int(running)
#                 BACKEND_METRICS[backend_url]["waiting"] = int(waiting)
#                 # Trata o N/A do vLLM
#                 kv_val = 0.0 if kv_cache == "N/A" else float(kv_cache)
#                 BACKEND_METRICS[backend_url]["kv_cache"] = kv_val
#                 BACKEND_METRICS[backend_url]["last_updated"] = time.time()

# #------------------------------------ Fim da atualização ------------------------------------

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Start monitoring threads when app starts."""
#     print("[INFO] Starting log monitoring threads...")
#     for backend_url, path in BACKEND_LOGS.items():
#         t = threading.Thread(target=monitor_logs, args=(backend_url, path), daemon=True)
#         t.start()
#     yield
#     print("[INFO] Shutting down proxy...")


# app = FastAPI(lifespan=lifespan)


# def resolve_model_candidates(model_name: str):
#     if model_name in MODEL_ROUTES:
#         return MODEL_ROUTES[model_name]

#     if model_name.startswith("hosted_vllm/"):
#         base_name = model_name.removeprefix("hosted_vllm/")
#         if base_name in MODEL_ROUTES:
#             return MODEL_ROUTES[base_name]

#     prefixed_name = f"hosted_vllm/{model_name}"
#     if prefixed_name in MODEL_ROUTES:
#         return MODEL_ROUTES[prefixed_name]

#     return None


# def print_backend_status():
#     """Print formatted metrics for all backends."""
#     with metrics_lock:
#         print("\n[STATUS] Current backend load:")
#         print("-" * 70)
#         for url, metrics in BACKEND_METRICS.items():
#             port = url.split(":")[-1]
#             running = metrics["running"]
#             waiting = metrics["waiting"]
#             kv = metrics["kv_cache"]
#             updated_ago = time.time() - metrics["last_updated"]
#             print(
#                 f"  • Port {port}: running={running:2d}, waiting={waiting:2d}, "
#                 f"KV={kv:5.1f}%, updated {updated_ago:4.1f}s ago"
#             )
#         print("-" * 70 + "\n")


# @app.get("/status")
# async def get_status():
#     """Return current backend metrics for visualization."""
#     with metrics_lock:
#         return BACKEND_METRICS


# # @app.post("/v1/chat/completions")
# # async def proxy_chat(request: Request):
# #     global monitoring_active, first_request_time

# #     start_time = time.time()
# #     body = await request.json()
# #     model = body.get("model")

# #     temperature = body.get("temperature", 0.0)
# #     # Ativa o monitoramento somente quando chega a primeira requisição
# #     if not monitoring_active:
# #         monitoring_active = True
# #         first_request_time = start_time
# #         print(
# #             f"[METRICS] Monitoring started at {datetime.now().isoformat()} "
# #             "(first request received)"
# #         )

# #     if model not in MODEL_ROUTES:
# #         return {"error": f"Unknown model: {model}"}

# #     candidates = MODEL_ROUTES[model]

# #     with metrics_lock:
# #         current_metrics = {url: BACKEND_METRICS[url].copy() for url in candidates}

# #     print(f"\n[REQUEST] Received new request for model: {model} | Temperature: {temperature}")
# #     print_backend_status()

# #     now = time.time()
# #     valid_backends = [
# #         url for url in candidates
# #         if now - current_metrics[url].get("last_updated", 0) < 60
# #     ]


# #     if not valid_backends:
# #         best_backend = candidates[0]
# #         print(f"[WARN] No valid metrics for {model}, fallback to {best_backend}")
# #     else:
# #         best_backend = min(
# #             valid_backends,
# #             key=lambda url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
# #         )
    

# #     total_loads = {
# #         url: current_metrics[url]["running"] + current_metrics[url]["waiting"]
# #         for url in valid_backends
# #     }

# #     print(f"[INFO] Load summary: {total_loads}")
# #     print(f"[INFO] Selected backend for {model}: {best_backend}\n")

# #     target_url = f"{best_backend}/v1/chat/completions"
    
# #     try:
# #         async with httpx.AsyncClient(timeout=36000.0) as client:
# #             resp = await client.post(target_url, json=body)

# #         latency = time.time() - start_time
# #         # log_latency(latency, best_backend)  # salva no CSV
# #         print(f"[METRIC] E2E latency = {latency:.3f}s | Backend = {best_backend}")

# #         return resp.json()
# #     except httpx.RequestError as e:
# #         latency = time.time() - start_time
# #         # log_latency(latency, best_backend)
# #         print(f"[ERROR] Failed request ({latency:.3f}s): {e}")
# #         return {"error": f"Backend unavailable: {best_backend}"}

# # ------------------- Script atualizado para logs mais detalhados e tratamento de erros -------------------
# @app.post("/v1/chat/completions")
# async def proxy_chat(request: Request):
#     global monitoring_active, first_request_time

#     start_time = time.time()
#     body = await request.json()
#     model = body.get("model")
#     temperature = body.get("temperature", 0.0)

#     # 1. Ativa monitoramento na primeira requisição
#     if not monitoring_active:
#         monitoring_active = True
#         first_request_time = start_time
#         print(f"[METRICS] Monitoramento iniciado em {datetime.now().isoformat()}")

#     # 2. Verifica se o modelo existe nas rotas
#     candidates = resolve_model_candidates(model)
#     if candidates is None:
#         print(f"[ERROR] Modelo desconhecido solicitado: {model}")
#         return {"error": f"Unknown model: {model}"}
#     now = time.time()

#     # 3. Seleção de Backend (Lógica de Load Balance)
#     valid_backends = []
#     if len(candidates) == 1:
#         # Se só tem um (ex: Llama na 8005), vai direto nele
#         best_backend = candidates[0]
#         valid_backends = [best_backend]
#     else:
#         # Se tem vários, filtra os que estão com métricas "frescas" (90s)
#         valid_backends = [
#             url for url in candidates
#             if (now - BACKEND_METRICS[url].get("last_updated", 0)) < 90
#         ]
        
#         if not valid_backends:
#             # Fallback se ninguém estiver logando métricas
#             best_backend = candidates[0]
#             valid_backends = [best_backend]
#             print(f"[WARN] Sem métricas frescas para {model}, usando fallback: {best_backend}")
#         else:
#             # Escolhe o que tem menos (running + waiting)
#             with metrics_lock:
#                 best_backend = min(
#                     valid_backends,
#                     key=lambda url: BACKEND_METRICS[url]["running"] + BACKEND_METRICS[url]["waiting"]
#                 )

#     # 4. Logs de depuração
#     with metrics_lock:
#         current_loads = {
#             url: f"R:{BACKEND_METRICS[url]['running']} W:{BACKEND_METRICS[url]['waiting']}"
#             for url in valid_backends
#         }

#     print(f"\n[REQUEST] Model: {model} | Temp: {temperature}")
#     print(f"[INFO] Load status for candidates: {current_loads}")
#     print(f"[INFO] Selected backend: {best_backend}")

#     # 5. Encaminhamento da requisição
#     target_url = f"{best_backend}/v1/chat/completions"
#     try:
#         async with httpx.AsyncClient(timeout=36000.0) as client:
#             resp = await client.post(target_url, json=body)
            
#         latency = time.time() - start_time
#         print(f"[METRIC] Latency = {latency:.3f}s | Backend = {best_backend}")
#         return resp.json()

#     except httpx.RequestError as e:
#         latency = time.time() - start_time
#         print(f"[ERROR] Falha na requisição ({latency:.3f}s): {e}")
#         return {"error": f"Backend {best_backend} indisponível"}

# # -------------------- Script atualizado para logs mais detalhados e tratamento de erros -------------------

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



## ALETERADO PARA TESTES DA AXIA--- 

import os
import re
import threading
import time
import csv
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import uvicorn
from threading import Lock
from contextlib import asynccontextmanager

LOG_DIR = os.environ.get("LOG_DIR", "./var/logs")
print(f"[INFO] LOG_DIR={LOG_DIR}")


MODEL_ROUTES = {
    # Mapeamento do MODEL1 (Llama) na porta 8005
    "meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8005"],
    "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct": ["http://localhost:8005"],

    #  Mapeamento do MODEL2 (Llama) na porta 8006
    "meta-llama/Llama-3.3-70B-Instruct": ["http://localhost:8006"],
    "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct": ["http://localhost:8006"],

    # # Mapeamento do MODEL3 (Qwen) na porta 8007
    # "Qwen/Qwen3-235B-A22B": ["http://localhost:8007"],
    # "hosted_vllm/Qwen/Qwen3-235B-A22B": ["http://localhost:8007"],

    # Mapeamento do MODEL4 (gptoss) na porta 8008
    "openai/gpt-oss-120b": ["http://localhost:8008"],
    "hosted_vllm/openai/gpt-oss-120b": ["http://localhost:8008"],
}

BACKEND_LOGS = {
    "http://localhost:8005": os.path.join(LOG_DIR, "saida_VLLM_8005.txt"),
    "http://localhost:8006": os.path.join(LOG_DIR, "saida_VLLM_8006.txt"),
    # "http://localhost:8007": os.path.join(LOG_DIR, "saida_VLLM_8007.txt"),
    "http://localhost:8008": os.path.join(LOG_DIR, "saida_VLLM_8008.txt"),
 }

# ------------------------------- Métricas e Padrões ------------------------------------
BACKEND_METRICS = {
    url: {
        "running": 0, 
        "waiting": 0, 
        "kv_cache": 0.0, 
        "last_updated": time.time()
    }
    for url in BACKEND_LOGS
}

LOG_PATTERN = re.compile(
    r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs,.*?(?:GPU KV cache usage:|KV cache usage:)\s*([0-9.]+|N/A)%"
)
#--------------------------------------------------------------------------------------------

metrics_lock = Lock()
# --- Métricas globais ---
monitoring_active = False
first_request_time = None


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
        print(f"[WARN] Log file {path} not found: {path}")
        while True:
            time.sleep(5)


def monitor_logs(backend_url, log_path):
    """Thread que monitora os logs e atualiza as métricas."""
    print(f"[DEBUG] Iniciando tail no log: {log_path} para o backend: {backend_url}")
    for line in tail_log_file(log_path, LOG_PATTERN):
        match = LOG_PATTERN.search(line)
        if match:
            running, waiting, kv_cache = match.groups()
            with metrics_lock:
                BACKEND_METRICS[backend_url]["running"] = int(running)
                BACKEND_METRICS[backend_url]["waiting"] = int(waiting)
                # Trata o N/A do vLLM
                kv_val = 0.0 if kv_cache == "N/A" else float(kv_cache)
                BACKEND_METRICS[backend_url]["kv_cache"] = kv_val
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


def resolve_model_candidates(model_name: str):
    if model_name in MODEL_ROUTES:
        return MODEL_ROUTES[model_name]

    if model_name.startswith("hosted_vllm/"):
        base_name = model_name.removeprefix("hosted_vllm/")
        if base_name in MODEL_ROUTES:
            return MODEL_ROUTES[base_name]

    prefixed_name = f"hosted_vllm/{model_name}"
    if prefixed_name in MODEL_ROUTES:
        return MODEL_ROUTES[prefixed_name]

    return None


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


@app.get("/health")
async def health():
    return {"status": "ok", "backends": sorted(BACKEND_LOGS)}


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

    # 1. Ativa monitoramento na primeira requisição
    if not monitoring_active:
        monitoring_active = True
        first_request_time = start_time
        print(f"[METRICS] Monitoramento iniciado em {datetime.now().isoformat()}")

    # 2. Verifica se o modelo existe nas rotas
    candidates = resolve_model_candidates(model)
    if candidates is None:
        print(f"[ERROR] Modelo desconhecido solicitado: {model}")
        return {"error": f"Unknown model: {model}"}
    now = time.time()

    # 3. Seleção de Backend (Lógica de Load Balance)
    valid_backends = []
    if len(candidates) == 1:
        # Se só tem uma rota associada (como as nossas rotas atuais), vai direto nela
        best_backend = candidates[0]
        valid_backends = [best_backend]
    else:
        # Se tivessem múltiplos nós para o mesmo modelo, filtra os ativos (90s)
        valid_backends = [
            url for url in candidates
            if (now - BACKEND_METRICS[url].get("last_updated", 0)) < 90
        ]
        
        if not valid_backends:
            best_backend = candidates[0]
            valid_backends = [best_backend]
            print(f"[WARN] Sem métricas frescas para {model}, usando fallback: {best_backend}")
        else:
            with metrics_lock:
                best_backend = min(
                    valid_backends,
                    key=lambda url: BACKEND_METRICS[url]["running"] + BACKEND_METRICS[url]["waiting"]
                )

    # 4. Logs de depuração
    with metrics_lock:
        current_loads = {
            url: f"R:{BACKEND_METRICS[url]['running']} W:{BACKEND_METRICS[url]['waiting']}"
            for url in valid_backends
        }

    print(f"\n[REQUEST] Model: {model} | Temp: {temperature}")
    print(f"[INFO] Load status for candidates: {current_loads}")
    print(f"[INFO] Selected backend: {best_backend}")

    # 5. Encaminhamento da requisição
    target_url = f"{best_backend}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=36000.0) as client:
            resp = await client.post(target_url, json=body)
            
        latency = time.time() - start_time
        print(f"[METRIC] Latency = {latency:.3f}s | Backend = {best_backend}")
        return resp.json()

    except httpx.RequestError as e:
        latency = time.time() - start_time
        print(f"[ERROR] Falha na requisição ({latency:.3f}s): {e}")
        return {"error": f"Backend {best_backend} indisponível"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



