#!/usr/bin/env python3
"""
Componente A do projeto Masterfile — EXTRATOR (externo, não-invasivo).

Converte um `ExecutionStats` (dict do model_dump) nos CSVs por run:

    runs/<run_id>/run.json            metadados + agregados
    runs/<run_id>/observations.csv    1 linha por (operador x registro) na amostragem
    runs/<run_id>/final_exec.csv      1 linha por (operador x registro) na execução final
    runs/<run_id>/operators.csv       1 linha por full_op_id visto nesta run

DUAS FORMAS DE USO
------------------
1) Automática, de dentro do cuad-demo1.py (recomendada). Não lê arquivo nenhum:

       from masterfile.extract import allocate_run_dir, extract_run

       # no INICIO da run:
       run_dir, run_id = allocate_run_dir(exp_name)

       # no FIM da run:
       extract_run(execution_stats_dict, run_dir, run_id, config=..., metrics=...)

2) CLI, para reprocessar arquivos historicos:

       python extract.py --stats opt-profiling-data/<exp>-stats.json --verify

FILTRAGEM
---------
Somente as chaves listadas em *_COLUMNS sao lidas. Tudo o mais e' descartado —
em especial `answer` (saida bruta do LLM, maior fonte de volume no stats.json),
`plan_str` dos planos sentinela, `input_fields` e `generated_fields`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any

# =============== NEW FEATURE ===============
# Especificacao de metricas por benchmark (CUAD: f1, BioDEX: rp@5).
try:
    from .benchmarks import get_spec, normalize_metrics          # import como pacote (masterfile.extract)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmarks import get_spec, normalize_metrics           # python masterfile/extract.py
# =============== END NEW FEATURE ===============

# ---------------------------------------------------------------------------
# Versão do schema do masterfile. INCREMENTAR ao adicionar/remover colunas.
# ---------------------------------------------------------------------------
# =============== NEW FEATURE ===============
# v2: run.json ganha quality, quality_metric e as metricas especificas do benchmark
# (ver masterfile/benchmarks.py). Colunas dos CSVs inalteradas.
SCHEMA_VERSION = 2
# (original)
# SCHEMA_VERSION = 1
# =============== END NEW FEATURE ===============

# ---------------------------------------------------------------------------
# Diretórios padrão. Resolvidos em relação a ESTE arquivo, não ao diretório de
# trabalho: assim os artefatos ficam sempre dentro de masterfile/, seja qual for
# o CWD de onde o script do benchmark foi lançado.
#
#   masterfile/runs/<run_id>/   diretórios por run
#   masterfile/out/             masterfile-*.csv consolidados
# ---------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(PKG_DIR, "runs")
DEFAULT_OUT_DIR = os.path.join(PKG_DIR, "out")


# ---------------------------------------------------------------------------
# Colunas. A ordem é fixa: é ela que garante CSVs estáveis entre runs.
# ---------------------------------------------------------------------------

OBSERVATIONS_COLUMNS = [
    "run_id",
    "sentinel_plan_id",
    "unique_logical_op_id",
    "full_op_id",
    "logical_op_id",
    "physical_op_id",
    "op_name",
    "iteration",          # None até o hook B1
    "obs_index",          # posição na record_op_stats_lst (proxy monotônico)
    "record_id",
    # as quatro métricas do cost model
    "quality",
    "cost_per_record",
    "time_per_record",
    "passed_operator",
    # tokens e chamadas (permitem recomputar custo)
    "model_name",
    "total_input_tokens",
    "total_output_tokens",
    "total_input_cost",
    "total_output_cost",
    "total_llm_calls",
    "total_embedding_llm_calls",
    "llm_call_duration_secs",
    "fn_call_duration_secs",
    # falhas
    "failed_convert",
    "error_type",         # None até o hook B3
    # linhagem (necessária para reproduzir a dedup e o cálculo de selectivity)
    "record_parent_ids",
    "record_source_indices",
    "plan_id",
]

FINAL_EXEC_COLUMNS = [
    "run_id",
    "plan_id",
    "unique_full_op_id",
    "full_op_id",
    "logical_op_id",
    "physical_op_id",
    "op_name",
    "obs_index",
    "record_id",
    "quality",            # normalmente vazio: não há validator na execução final
    "cost_per_record",
    "time_per_record",
    "passed_operator",
    "model_name",
    "total_input_tokens",
    "total_output_tokens",
    "total_input_cost",
    "total_output_cost",
    "total_llm_calls",
    "total_embedding_llm_calls",
    "llm_call_duration_secs",
    "fn_call_duration_secs",
    "failed_convert",
    "error_type",
    "record_parent_ids",
    "record_source_indices",
]

OPERATORS_COLUMNS = [
    "full_op_id",
    "logical_op_id",
    "physical_op_id",
    "op_name",
    "op_details_json",    # == get_id_params(); é o próprio insumo do hash de identidade
    "phase",              # sampling | final | both
    "first_seen_run_id",
]


# ---------------------------------------------------------------------------
# Alocação de diretório de run — SEM SOBRESCRIÇÃO
# ---------------------------------------------------------------------------

def allocate_run_dir(exp_name: str, root: str | None = None) -> tuple[str, str]:
    """
    Reserva um diretório novo para a run e devolve (run_dir, run_id).

    Se `runs/<exp_name>` já existe, tenta `<exp_name>__002`, `__003`, ...
    Nenhuma run é sobrescrita, mesmo que --exp-name se repita.

    Deve ser chamada UMA VEZ, no INÍCIO da run. O diretório devolvido é o mesmo
    usado depois por extract_run(); alocar duas vezes separaria o run_config.json
    dos CSVs.

    `root` default: masterfile/runs/
    """
    root = root or DEFAULT_RUNS_DIR
    os.makedirs(root, exist_ok=True)

    candidate = exp_name
    suffix = 1
    while os.path.exists(os.path.join(root, candidate)):
        suffix += 1
        candidate = f"{exp_name}__{suffix:03d}"

    run_dir = os.path.join(root, candidate)
    os.makedirs(run_dir)  # sem exist_ok: falha alto se houver corrida
    return run_dir, candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_full_op_id(full_op_id: str) -> tuple[str, str]:
    """
    full_op_id tem a forma '<logical_op_id>-<physical_op_id>'. Ambas as metades
    são hexdigests sha256 truncados; o separador é o ÚLTIMO hífen.
    """
    if not full_op_id or "-" not in full_op_id:
        return ("", full_op_id or "")
    logical, _, physical = full_op_id.rpartition("-")
    return (logical, physical)


def jdump(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


# =============== NEW FEATURE ===============
# Aceita tambem o sufixo -profiling.json dos scripts do BioDEX.
def exp_name_from_stats_path(path: str) -> str:
    # CUAD salva <exp>-stats.json; os scripts do BioDEX salvam <exp>-profiling.json
    return re.sub(r"-(stats|profiling)\.json$", "", os.path.basename(path))
# (original)
# def exp_name_from_stats_path(path: str) -> str:
#     return re.sub(r"-stats\.json$", "", os.path.basename(path))
# =============== END NEW FEATURE ===============


def parse_config_from_exp_name(exp_name: str) -> dict:
    """
    Fallback para runs históricas: extrai o que der do nome do experimento.
    Nunca inventa valores — o que não casar fica None.
    """
    cfg: dict[str, Any] = {}
    for key, pat in {
        "k": r"-k(\d+)",
        "j": r"-j(\d+)",
        "sample_budget": r"-budget(\d+)",
        "seed": r"-seed(\d+)",
    }.items():
        m = re.search(pat, exp_name)
        cfg[key] = int(m.group(1)) if m else None

    cfg["sentinel_strategy"] = None
    for strategy in ("cxtbay", "mab", "bay", "all"):
        if re.search(rf"(^|-){strategy}(-|$)", exp_name):
            cfg["sentinel_strategy"] = strategy.upper()
            break

    cfg["benchmark"] = None
    for bench in ("cuad", "biodex", "mmqa"):
        if exp_name.startswith(bench):
            cfg["benchmark"] = bench
            break

    return cfg


def write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def _record_row(rec: dict, base: dict, obs_index: int, columns: list[str]) -> dict:
    """Converte um RecordOpStats (dict do model_dump) numa linha de CSV."""
    row = dict(base)
    row["obs_index"] = obs_index
    for key in ("record_id", "quality", "cost_per_record", "time_per_record",
                "passed_operator", "model_name", "total_input_tokens",
                "total_output_tokens", "total_input_cost", "total_output_cost",
                "total_llm_calls", "total_embedding_llm_calls",
                "llm_call_duration_secs", "fn_call_duration_secs",
                "failed_convert", "plan_id"):
        row[key] = rec.get(key)

    row["record_parent_ids"] = jdump(rec.get("record_parent_ids"))
    row["record_source_indices"] = jdump(rec.get("record_source_indices"))

    # Campos que só existem após os hooks do Componente B. Aceita tanto o campo
    # no nível raiz (opção A: campo novo em RecordOpStats) quanto dentro de
    # op_details (opção B: sem modificar o modelo pydantic).
    op_details = rec.get("op_details") or {}
    row["iteration"] = rec.get("sampling_iteration", op_details.get("sampling_iteration"))
    row["error_type"] = rec.get("error_type", op_details.get("error_type"))

    return {c: row.get(c) for c in columns}


def extract_observations(stats: dict, run_id: str) -> tuple[list[dict], dict]:
    """sentinel_plan_stats[plan_id].operator_stats[unique_logical_op_id][full_op_id]"""
    rows: list[dict] = []
    operators: dict[str, dict] = {}

    for sentinel_plan_id, plan_stats in (stats.get("sentinel_plan_stats") or {}).items():
        for unique_logical_op_id, by_full_op_id in (plan_stats.get("operator_stats") or {}).items():
            if not isinstance(by_full_op_id, dict):
                continue
            for full_op_id, op_stats in by_full_op_id.items():
                if not isinstance(op_stats, dict):
                    continue
                logical_op_id, physical_op_id = split_full_op_id(full_op_id)

                operators.setdefault(full_op_id, {
                    "full_op_id": full_op_id,
                    "logical_op_id": logical_op_id,
                    "physical_op_id": physical_op_id,
                    "op_name": op_stats.get("op_name"),
                    "op_details_json": jdump(op_stats.get("op_details") or {}),
                    "phase": "sampling",
                    "first_seen_run_id": run_id,
                })

                base = {
                    "run_id": run_id,
                    "sentinel_plan_id": sentinel_plan_id,
                    "unique_logical_op_id": unique_logical_op_id,
                    "full_op_id": full_op_id,
                    "logical_op_id": logical_op_id,
                    "physical_op_id": physical_op_id,
                    "op_name": op_stats.get("op_name"),
                }
                for i, rec in enumerate(op_stats.get("record_op_stats_lst") or []):
                    rows.append(_record_row(rec, base, i, OBSERVATIONS_COLUMNS))

    return rows, operators


def extract_final_exec(stats: dict, run_id: str, operators: dict) -> list[dict]:
    """plan_stats[plan_id].operator_stats[unique_full_op_id]  (um nível a menos)"""
    rows: list[dict] = []

    for plan_id, plan_stats in (stats.get("plan_stats") or {}).items():
        for unique_full_op_id, op_stats in (plan_stats.get("operator_stats") or {}).items():
            if not isinstance(op_stats, dict):
                continue
            full_op_id = op_stats.get("full_op_id") or ""
            logical_op_id, physical_op_id = split_full_op_id(full_op_id)

            if full_op_id in operators:
                operators[full_op_id]["phase"] = "both"
            else:
                operators[full_op_id] = {
                    "full_op_id": full_op_id,
                    "logical_op_id": logical_op_id,
                    "physical_op_id": physical_op_id,
                    "op_name": op_stats.get("op_name"),
                    "op_details_json": jdump(op_stats.get("op_details") or {}),
                    "phase": "final",
                    "first_seen_run_id": run_id,
                }

            base = {
                "run_id": run_id,
                "plan_id": plan_id,
                "unique_full_op_id": unique_full_op_id,
                "full_op_id": full_op_id,
                "logical_op_id": logical_op_id,
                "physical_op_id": physical_op_id,
                "op_name": op_stats.get("op_name"),
            }
            for i, rec in enumerate(op_stats.get("record_op_stats_lst") or []):
                rows.append(_record_row(rec, base, i, FINAL_EXEC_COLUMNS))

    return rows


def build_run_json(stats: dict, config: dict, run_id: str,
                   observations: list[dict], final_exec: list[dict],
                   operators: dict, metrics: dict | None) -> dict:
    final_plans = stats.get("plan_stats") or {}
    final_plan_ids = list(final_plans.keys())
    first_plan_id = final_plan_ids[0] if final_plan_ids else None

    run = {
        "run_id": run_id,
        "execution_id": stats.get("execution_id"),
        "schema_version": SCHEMA_VERSION,
        **dict(config),
        "optimization_time": stats.get("optimization_time"),
        "optimization_cost": stats.get("optimization_cost"),
        "plan_execution_time": stats.get("plan_execution_time"),
        "plan_execution_cost": stats.get("plan_execution_cost"),
        "total_execution_time": stats.get("total_execution_time"),
        "total_execution_cost": stats.get("total_execution_cost"),
        "total_input_tokens": stats.get("total_input_tokens"),
        "total_output_tokens": stats.get("total_output_tokens"),
        "final_plan_id": first_plan_id,
        "final_plan_str": (stats.get("plan_strs") or {}).get(first_plan_id) if first_plan_id else None,
        "n_sentinel_plans": len(stats.get("sentinel_plan_stats") or {}),
        "n_observations": len(observations),
        "n_final_exec_records": len(final_exec),
        "n_distinct_operators": len(operators),
        "n_distinct_operators_sampled": sum(
            1 for o in operators.values() if o["phase"] in ("sampling", "both")
        ),
    }

    if metrics:
        for key in ("precision", "recall", "f1"):
            if key in metrics:
                run[key] = metrics[key]

    # =============== NEW FEATURE ===============
    # Qualidade padronizada entre benchmarks (antes o RP@5 do BioDEX era descartado).
    # quality / quality_metric + metricas especificas do benchmark.
    # Mantem precision/recall/f1 acima para compatibilidade com runs antigas do CUAD.
    run.update(normalize_metrics(config.get("benchmark"), metrics))
    # =============== END NEW FEATURE ===============

    # garante que run_id nunca seja sobrescrito por uma chave homônima do config
    run["run_id"] = run_id
    return run


# ---------------------------------------------------------------------------
# Testes de aceitação embutidos
# ---------------------------------------------------------------------------

def verify(run: dict, observations: list[dict], operators: dict) -> list[str]:
    """Lista de alertas. Vazia == nenhum problema detectado."""
    warnings: list[str] = []

    # Aceitação 4 — conservação de custo
    obs_cost = sum(float(r["cost_per_record"] or 0.0) for r in observations)
    opt_cost = run.get("optimization_cost")
    if opt_cost:
        delta = abs(obs_cost - float(opt_cost)) / float(opt_cost)
        if delta > 0.01:
            direction = "MENOR" if obs_cost < float(opt_cost) else "MAIOR"
            warnings.append(
                f"[conservacao] soma de cost_per_record ({obs_cost:.6f}) e' {direction} que "
                f"optimization_cost ({float(opt_cost):.6f}); diferenca de {delta:.1%}. "
                f"MENOR e' esperado (optimization_cost inclui o validator). "
                f"MAIOR indica dupla contagem."
            )

    # Aceitação 5 — contagem vs. budget, k e j
    # =============== NEW FEATURE ===============
    # Budget: a 1a rodada custa k*j POR operador (BioDEX tem 3). Aceita tambem
    # sentinel_execution_strategy (nome do argparse nas runs ao vivo).
    k, j, budget = run.get("k"), run.get("j"), run.get("sample_budget")
    strategy = (run.get("sentinel_strategy") or run.get("sentinel_execution_strategy") or "").upper()
    spec = get_spec(run.get("benchmark"))
    n_ops = run.get("num_semantic_ops") or (spec.num_semantic_ops if spec else 1)
    if strategy != "ALL" and all(isinstance(v, int) for v in (k, j, budget, n_ops)):
        first_round = k * j * n_ops
        if first_round > budget:
            warnings.append(
                f"[budget] k*j*ops = {k}*{j}*{n_ops} = {first_round} > sample_budget = {budget}. "
                f"A primeira rodada ja' excede o budget: o laco encerra apos uma iteracao, "
                f"update_frontier roda no maximo uma vez, e as estrategias ficam indistinguiveis."
            )
    # (original)
    # k, j, budget = run.get("k"), run.get("j"), run.get("sample_budget")
    # strategy = (run.get("sentinel_strategy") or "").upper()
    # if strategy != "ALL" and all(isinstance(v, int) for v in (k, j, budget)):
    #     if k * j > budget:
    #         warnings.append(
    #             f"[budget] k*j = {k*j} > sample_budget = {budget}. A primeira rodada ja' "
    #             f"excede o budget: o laco encerra apos uma iteracao, update_frontier roda no "
    #             f"maximo uma vez, e as estrategias ficam indistinguiveis."
    #         )
    # =============== END NEW FEATURE ===============

    # Identidade
    seen: dict[str, str] = {}
    for full_op_id, op in operators.items():
        details = op["op_details_json"]
        if full_op_id in seen and seen[full_op_id] != details:
            warnings.append(f"[identidade] full_op_id {full_op_id} com op_details divergentes.")
        seen[full_op_id] = details

    # Cobertura
    observed = {r["full_op_id"] for r in observations}
    never = [fid for fid, op in operators.items()
             if op["phase"] in ("sampling", "both") and fid not in observed]
    if never:
        warnings.append(
            f"[cobertura] {len(never)} operadores no espaco amostrado sem nenhuma observacao. "
            f"Ate o hook B3, falhas e timeouts sao indistinguiveis de 'nunca agendado'."
        )

    return warnings


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract_run(stats: dict,
                run_dir: str,
                run_id: str,
                config: dict | None = None,
                metrics: dict | None = None,
                verify_run: bool = True,
                quiet: bool = False) -> dict:
    """
    Escreve run.json + os três CSVs em `run_dir` a partir do dict de ExecutionStats.

    Não lê arquivo algum: `stats` é o retorno de `execution_stats.to_json()`.
    Devolve o dict do run.json.
    """
    config = dict(config or {})
    os.makedirs(run_dir, exist_ok=True)

    observations, operators = extract_observations(stats, run_id)
    final_exec = extract_final_exec(stats, run_id, operators)
    run = build_run_json(stats, config, run_id, observations, final_exec, operators, metrics)

    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False, default=str)

    write_csv(os.path.join(run_dir, "observations.csv"), OBSERVATIONS_COLUMNS, observations)
    write_csv(os.path.join(run_dir, "final_exec.csv"), FINAL_EXEC_COLUMNS, final_exec)
    write_csv(os.path.join(run_dir, "operators.csv"), OPERATORS_COLUMNS, list(operators.values()))

    if not quiet:
        print(f"[masterfile] run_id ................ {run_id}")
        print(f"[masterfile] observacoes ........... {len(observations)}")
        print(f"[masterfile] registros exec. final . {len(final_exec)}")
        print(f"[masterfile] operadores distintos .. {len(operators)}")
        print(f"[masterfile] saida ................. {run_dir}/")

    if verify_run:
        warnings = verify(run, observations, operators)
        if not quiet:
            if warnings:
                print(f"[masterfile] VERIFICACAO: {len(warnings)} alerta(s)")
                for w in warnings:
                    print(f"[masterfile]   - {w}")
            else:
                print("[masterfile] VERIFICACAO: nenhum alerta.")
        with open(os.path.join(run_dir, "verify.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(warnings) if warnings else "OK: nenhum alerta.\n")

    return run


# ---------------------------------------------------------------------------
# CLI (apenas para reprocessar arquivos históricos)
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extrator do masterfile. Para runs novas, prefira chamar "
                    "allocate_run_dir()/extract_run() de dentro do script do benchmark."
    )
    ap.add_argument("--stats", required=True, help="caminho do <exp_name>-stats.json")
    ap.add_argument("--metrics", default=None, help="caminho do <exp_name>-metrics.json")
    ap.add_argument("--config", default=None, help="sidecar run_config.json")
    ap.add_argument("--out", default=None,
                    help=f"diretorio raiz das runs (default: {DEFAULT_RUNS_DIR})")
    ap.add_argument("--benchmark", default=None, help="cuad | biodex | mmqa")
    ap.add_argument("--campaign", default=None, choices=["census", "behavior"])
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.stats):
        print(f"ERRO: arquivo nao encontrado: {args.stats}", file=sys.stderr)
        return 1

    exp_name = exp_name_from_stats_path(args.stats)

    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)

    config = parse_config_from_exp_name(exp_name)
    config["exp_name"] = exp_name
    config["config_source"] = "filename"
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            config.update(json.load(f))
        config["config_source"] = "sidecar"
    if args.benchmark:
        config["benchmark"] = args.benchmark
    if args.campaign:
        config["campaign"] = args.campaign

    metrics = None
    # =============== NEW FEATURE ===============
    # Metrics ao lado de -stats.json (CUAD) ou -profiling.json (BioDEX).
    metrics_path = args.metrics or re.sub(r"-(stats|profiling)\.json$", "-metrics.json", args.stats)
    # (original)
    # metrics_path = args.metrics or args.stats.replace("-stats.json", "-metrics.json")
    # =============== END NEW FEATURE ===============
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    run_dir, run_id = allocate_run_dir(exp_name, args.out)
    extract_run(stats, run_dir, run_id, config=config, metrics=metrics,
                verify_run=args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())