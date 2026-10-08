"""
Diagnóstico rápido do pipeline CUAD.

Roda no mesmo container/diretório do cuad-demo1.py:
    python diag_cuad.py

Responde a uma única pergunta: os dados do CUAD chegaram ao container e
quantos contratos o pipeline realmente enxerga?
"""

import inspect
import sys

import numpy as np

try:
    from cuad_data_loader import load_cuad_data
except Exception as exc:  # noqa: BLE001
    print(f"[FALHA] não foi possível importar cuad_data_loader: {type(exc).__name__}: {exc}")
    sys.exit(1)

print(f"[OK] cuad_data_loader: {inspect.getfile(load_cuad_data)}")


def build_contracts(dataset, num_contracts: int, seed: int = 42):
    """Mesma lógica de CUADDataset._construct_dataset, sem depender do palimpzest."""
    contract_titles = []
    for row in dataset:
        if row["title"] not in contract_titles:
            contract_titles.append(row["title"])

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(contract_titles)
    contract_titles = contract_titles[:num_contracts]

    contracts = []
    for title in contract_titles:
        contract_rows = [row for row in dataset if row["title"] == title]
        contracts.append({
            "contract_id": contract_rows[0]["id"],
            "title": title,
            "contract": contract_rows[0]["context"],
        })

    return contracts


problemas = []

for split, pedidos in (("train", 25), ("test", 100)):
    print(f"\n=== split: {split} (num_contracts pedido: {pedidos}) ===")

    try:
        dataset = load_cuad_data(split=split)
    except Exception as exc:  # noqa: BLE001
        print(f"[FALHA] load_cuad_data(split={split!r}) levantou {type(exc).__name__}: {exc}")
        problemas.append(f"{split}: load_cuad_data falhou")
        continue

    linhas = len(dataset)
    print(f"linhas retornadas: {linhas}")
    if linhas == 0:
        print("[FALHA] split vazio — os dados do CUAD não estão no container")
        problemas.append(f"{split}: 0 linhas")
        continue

    primeira = dataset[0]
    chaves = list(primeira.keys()) if hasattr(primeira, "keys") else type(primeira)
    print(f"chaves da primeira linha: {chaves}")

    for campo in ("title", "id", "context"):
        if campo not in primeira:
            print(f"[FALHA] campo obrigatório ausente: {campo!r}")
            problemas.append(f"{split}: campo {campo} ausente")

    contratos = build_contracts(dataset, pedidos)
    print(f"contratos construídos: {len(contratos)} (pedidos: {pedidos})")

    if len(contratos) < pedidos:
        print("[FALHA] menos contratos do que o pedido — __len__ devolve num_contracts "
              "e o palimpzest vai pedir índices que não existem")
        problemas.append(f"{split}: {len(contratos)} de {pedidos} contratos")

    if contratos:
        tamanhos = [len(c["contract"] or "") for c in contratos]
        print(f"tamanho do texto do contrato: min={min(tamanhos)} "
              f"mediana={int(np.median(tamanhos))} max={max(tamanhos)} caracteres")
        if min(tamanhos) == 0:
            print("[FALHA] há contrato com texto vazio")
            problemas.append(f"{split}: contrato com texto vazio")
        print(f"exemplo: contract_id={contratos[0]['contract_id']!r} "
              f"title={contratos[0]['title']!r}")

print("\n=== resultado ===")
if problemas:
    print("problemas encontrados:")
    for p in problemas:
        print(f"  - {p}")
    sys.exit(2)

print("dados OK: os splits têm contratos suficientes e com texto.")
print("se o scan continuar em 0/0, o problema está na integração com o palimpzest, "
      "não nos dados.")