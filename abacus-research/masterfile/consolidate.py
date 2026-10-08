#!/usr/bin/env python3
"""
Componente A do projeto Masterfile — CONSOLIDADOR.

Varre `runs/*/` e regenera os masterfiles do zero. É IDEMPOTENTE: rodar duas
vezes produz exatamente o mesmo resultado. Nunca faz append — sempre reconstrói.

    masterfile-<benchmark>.csv              observações (amostragem) + config da run
    masterfile-<benchmark>-final.csv        execução final + config da run
    masterfile-<benchmark>-runs.csv         1 linha por run
    masterfile-<benchmark>-operators.csv    dimensão global de operadores

O join com `operators` é feito aqui: `op_details_json` é explodido em colunas
`param_<chave>`, calculando a união das chaves vistas em todas as runs. Assim o
masterfile sai desnormalizado (arquivo único, como pedido) sem que os CSVs por
run paguem o custo de repetir hiperparâmetros em milhões de linhas.

Por padrão lê de `masterfile/runs/` e escreve em `masterfile/out/`.

Uso:
    python consolidate.py --benchmark cuad
    python consolidate.py --benchmark cuad --no-explode-params
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Diretórios padrão, resolvidos em relação a ESTE arquivo (não ao CWD).
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(PKG_DIR, "runs")
DEFAULT_OUT_DIR = os.path.join(PKG_DIR, "out")

# Colunas da run que são propagadas para as linhas de observação.
# Mantidas curtas de propósito: o resto fica em masterfile-<bench>-runs.csv,
# acessível por join em run_id.
RUN_COLUMNS_TO_PROPAGATE = [
    "benchmark",
    "campaign",
    "sentinel_strategy",
    "optimizer_strategy",
    "policy_class",
    "k",
    "j",
    "sample_budget",
    "seed",
    "mode",
    "schema_version",
    "prior_store_training",
    "prior_store_training_runs",
]


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_runs(runs_dir: str, benchmark: str | None) -> list[tuple[str, dict]]:
    """Retorna [(run_dir, run_json)] das runs válidas, ordenadas por run_id."""
    found = []
    if not os.path.isdir(runs_dir):
        return found
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        run_json_path = os.path.join(run_dir, "run.json")
        if not os.path.isfile(run_json_path):
            continue
        with open(run_json_path, encoding="utf-8") as f:
            run = json.load(f)
        if benchmark and run.get("benchmark") not in (None, benchmark):
            continue
        found.append((run_dir, run))
    return found


def union_columns(rows: list[dict], preferred: list[str]) -> list[str]:
    """Colunas na ordem preferida primeiro; o resto ordenado alfabeticamente."""
    seen = set()
    for row in rows:
        seen.update(row.keys())
    ordered = [c for c in preferred if c in seen]
    ordered += sorted(c for c in seen if c not in set(ordered))
    return ordered


def write_csv(path: str, rows: list[dict], preferred: list[str]) -> int:
    columns = union_columns(rows, preferred)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def build_operator_params(operator_rows: list[dict], explode: bool) -> dict[str, dict]:
    """
    Monta {full_op_id: {param_<chave>: valor, ...}} a partir de op_details_json.
    A união das chaves é calculada sobre todas as runs.
    """
    params_by_op: dict[str, dict] = {}
    if not explode:
        return params_by_op
    for row in operator_rows:
        full_op_id = row.get("full_op_id")
        if not full_op_id:
            continue
        try:
            details = json.loads(row.get("op_details_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            details = {}
        params_by_op.setdefault(full_op_id, {}).update(
            {f"param_{k}": v for k, v in details.items()}
        )
    return params_by_op


def main() -> int:
    ap = argparse.ArgumentParser(description="Consolidador do masterfile (idempotente).")
    ap.add_argument("--runs", default=DEFAULT_RUNS_DIR,
                    help=f"diretorio raiz das runs (default: {DEFAULT_RUNS_DIR})")
    ap.add_argument("--benchmark", required=True, help="cuad | biodex | mmqa")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help=f"diretorio de saida dos masterfiles (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--no-explode-params", action="store_true",
                    help="nao explodir op_details_json em colunas param_*")
    args = ap.parse_args()

    explode = not args.no_explode_params
    runs = discover_runs(args.runs, args.benchmark)
    if not runs:
        print(f"Nenhuma run encontrada em {args.runs}/ para o benchmark {args.benchmark}.")
        return 1

    all_runs: list[dict] = []
    all_operators: dict[str, dict] = {}
    operator_rows_raw: list[dict] = []
    all_obs: list[dict] = []
    all_final: list[dict] = []

    for run_dir, run in runs:
        run_id = run.get("run_id") or os.path.basename(run_dir)
        all_runs.append(run)

        op_rows = read_csv(os.path.join(run_dir, "operators.csv"))
        operator_rows_raw.extend(op_rows)
        for row in op_rows:
            fid = row.get("full_op_id")
            if not fid:
                continue
            if fid not in all_operators:
                all_operators[fid] = dict(row)
            elif all_operators[fid].get("phase") != row.get("phase"):
                all_operators[fid]["phase"] = "both"

        run_ctx = {c: run.get(c) for c in RUN_COLUMNS_TO_PROPAGATE if c in run}

        for row in read_csv(os.path.join(run_dir, "observations.csv")):
            row.setdefault("run_id", run_id)
            row.update(run_ctx)
            all_obs.append(row)

        for row in read_csv(os.path.join(run_dir, "final_exec.csv")):
            row.setdefault("run_id", run_id)
            row.update(run_ctx)
            all_final.append(row)

    # desnormaliza os hiperparâmetros nas tabelas de fato
    params_by_op = build_operator_params(operator_rows_raw, explode)
    if explode:
        for row in all_obs:
            row.update(params_by_op.get(row.get("full_op_id"), {}))
        for row in all_final:
            row.update(params_by_op.get(row.get("full_op_id"), {}))
        for fid, op in all_operators.items():
            op.update(params_by_op.get(fid, {}))

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, f"masterfile-{args.benchmark}")

    obs_pref = ["run_id", "campaign", "sentinel_strategy", "seed", "iteration",
                "full_op_id", "op_name", "record_id",
                "quality", "cost_per_record", "time_per_record", "passed_operator"]
    fin_pref = ["run_id", "campaign", "sentinel_strategy", "seed",
                "full_op_id", "op_name", "record_id",
                "cost_per_record", "time_per_record"]
    run_pref = ["run_id", "exp_name", "benchmark", "campaign", "sentinel_strategy",
                "optimizer_strategy", "policy_class", "k", "j", "sample_budget", "seed",
                "f1", "precision", "recall"]
    op_pref = ["full_op_id", "logical_op_id", "physical_op_id", "op_name", "phase"]

    n_obs = write_csv(f"{prefix}.csv", all_obs, obs_pref)
    n_fin = write_csv(f"{prefix}-final.csv", all_final, fin_pref)
    n_run = write_csv(f"{prefix}-runs.csv", all_runs, run_pref)
    n_op = write_csv(f"{prefix}-operators.csv", list(all_operators.values()), op_pref)

    print(f"runs consolidadas ..... {n_run}")
    print(f"observacoes ........... {n_obs}  -> {prefix}.csv")
    print(f"registros exec. final . {n_fin}  -> {prefix}-final.csv")
    print(f"operadores distintos .. {n_op}  -> {prefix}-operators.csv")

    # sanidade: duas runs com o mesmo execution_id sao extracoes DUPLICADAS do
    # mesmo stats.json. O contador de colisao de allocate_run_dir impede
    # sobrescricao, mas nao impede que a mesma run seja extraida duas vezes —
    # o que faria o masterfile contar tudo em dobro.
    by_exec: dict = {}
    for r in all_runs:
        exec_id = r.get("execution_id")
        if exec_id:
            by_exec.setdefault(exec_id, []).append(r.get("run_id"))
    dups = {e: ids for e, ids in by_exec.items() if len(ids) > 1}
    if dups:
        print("\nALERTA: extracoes DUPLICADAS detectadas (mesmo execution_id).")
        print("As observacoes estao sendo contadas em dobro. Apague os diretorios extras:")
        for exec_id, run_ids in dups.items():
            print(f"  execution_id {exec_id}: {run_ids}")

    # sanidade: schema_version misturado é sinal de que colunas mudaram no meio
    versions = {r.get("schema_version") for r in all_runs}
    if len(versions) > 1:
        print(f"\nALERTA: schema_version misturado entre runs: {sorted(versions)}. "
              f"Colunas podem estar inconsistentes no historico.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())