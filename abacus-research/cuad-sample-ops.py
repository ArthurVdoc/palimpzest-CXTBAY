#!/usr/bin/env python3
"""
cuad-sample-ops.py — Amostragem DIRIGIDA de operadores fisicos do CUAD.

Objetivo
--------
Garantir pelo menos N amostras (fase de sampling) de uma lista fixa de operadores
fisicos (target_ops_cuad.json), para os boxplots de qualidade/custo/latencia.

Como funciona (sem alterar a logica do Abacus)
----------------------------------------------
1. Monta a MESMA query e a MESMA configuracao do cuad-demo1.py (mesmos modelos,
   api_base e flags allow_*). Isso e' obrigatorio: o full_op_id e' um hash dos
   parametros do operador; com outra config os ids nao batem.
2. Gera o sentinel plan normal do Palimpzest (processor._create_sentinel_plan).
3. Filtra o operator_set do plano para manter SO os operadores alvo (o scan fica).
4. Executa a estrategia que ja existe no Palimpzest, AllSamplingExecutionStrategy
   ("all"): todo operador do plano roda em todo registro de treino. Mesma execucao
   (_execute_op_set), mesmo validator (_score_quality, F1 do CUAD) e mesmo
   SentinelPlanStats do MAB/CXTBAY -> medidas comparaveis.
   Com --num-records N, cada operador alvo recebe N amostras.
5. NAO executa o plano final nos 100 contratos de teste (desnecessario aqui).

Saidas
------
- masterfile/runs/<run_id>/            (run.json, observations.csv, operators.csv, ...)
                                        mesmo formato das runs normais (via MasterfileRun)
- masterfile/targeted_sampling/observations-cuad.csv
                                        ARQUIVO ACUMULADO so com as amostragens dirigidas
                                        (todas as runs deste script, sem o scan)
- masterfile/targeted_sampling/summary-cuad.csv
                                        amostras validas por operador alvo (regenerado)
- opt-profiling-data/<exp_name>-stats.json   (ExecutionStats bruto, como no cuad-demo1.py)

Pre-requisitos em tempo de execucao: vLLM (vllm-axia.sh) e scheduler.py rodando,
exatamente como para o cuad-demo1.py. O --dry-run NAO precisa deles.

Uso
---
    python cuad-sample-ops.py --dry-run            # confere se os 13 ids existem no plano
    python cuad-sample-ops.py --num-records 5      # roda a amostragem
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # cuad_data_loader e o pacote masterfile

# Configuracao identica ao cuad-demo1.py (linhas do QueryProcessorConfig).
# Se o cuad-demo1.py mudar de modelos, atualize aqui tambem.
AVAILABLE_MODELS = [
    "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    # "hosted_vllm/Qwen/Qwen3-235B-A22B",
    "hosted_vllm/openai/gpt-oss-120b",
]
DEFAULT_API_BASE = "http://localhost:8000/v1"   # scheduler.py (porta 8000)

TARGETED_DIRNAME = "targeted_sampling"


# ---------------------------------------------------------------------------
# Reuso do cuad-demo1.py sem copiar codigo (o __main__ dele e' protegido)
# ---------------------------------------------------------------------------
def load_cuad_demo():
    path = os.path.join(HERE, "cuad-demo1.py")
    spec = importlib.util.spec_from_file_location("cuad_demo1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Filtro do sentinel plan
# ---------------------------------------------------------------------------
def _iter_plan_nodes(plan):
    for subplan in plan.subplans:
        yield from _iter_plan_nodes(subplan)
    yield plan


def filter_sentinel_plan(plan, target_ids: set[str], verbose: bool = True) -> list[str]:
    """
    Mantem no plano apenas os operadores alvo (scans ficam intactos).
    Idempotente. Levanta SystemExit se algum alvo nao existir no plano.
    Devolve a lista de full_op_ids mantidos.
    """
    from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp

    kept: list[str] = []
    available: list[tuple[str, str]] = []
    for node in _iter_plan_nodes(plan):
        ops = node.operator_set
        if isinstance(ops[0], (ScanPhysicalOp, ContextScanOp)):
            continue
        available.extend((op.get_full_op_id(), op.op_name()) for op in ops)
        keep = [op for op in ops if op.get_full_op_id() in target_ids]
        if not keep:
            raise SystemExit(
                f"[sample-ops] ERRO: nenhum operador alvo no operador logico {node.logical_op_id}. "
                f"Um operador logico nao pode ficar vazio."
            )
        node.operator_set = keep
        kept.extend(op.get_full_op_id() for op in keep)

    missing = sorted(target_ids - set(kept))
    if missing:
        print("[sample-ops] Operadores disponiveis no sentinel plan:")
        for fid, name in sorted(available):
            print(f"    {fid}  {name}")
        raise SystemExit(
            f"[sample-ops] ERRO: {len(missing)} operador(es) alvo nao existem no sentinel plan: {missing}. "
            f"A configuracao (modelos/api_base/flags allow_*) provavelmente diverge da usada nas runs originais."
        )

    if verbose:
        print(f"[sample-ops] {len(kept)} operador(es) alvo encontrados no sentinel plan "
              f"(de {len(available)} no espaco de busca).")
    return kept


def make_strategy_class():
    from palimpzest.query.execution.all_sample_execution_strategy import AllSamplingExecutionStrategy

    class TargetedAllSamplingStrategy(AllSamplingExecutionStrategy):
        """AllSampling restrito a um conjunto de full_op_ids. Nao altera a execucao."""

        def __init__(self, target_full_op_ids, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_full_op_ids = set(target_full_op_ids)

        def execute_sentinel_plan(self, plan, train_dataset, validator):
            filter_sentinel_plan(plan, self.target_full_op_ids, verbose=False)
            return super().execute_sentinel_plan(plan, train_dataset, validator)

    return TargetedAllSamplingStrategy


# ---------------------------------------------------------------------------
# Arquivo acumulado em masterfile/targeted_sampling/
# ---------------------------------------------------------------------------
def _is_valid(row: dict) -> bool:
    """Mesmo criterio da tabela: descarta failed_convert verdadeiro."""
    return str(row.get("failed_convert", "")).strip().lower() not in ("true", "1")


def append_targeted_observations(run_dir: str, run_id: str, targets: list[dict]) -> str:
    """
    Copia as observacoes (sem scan) desta run para o arquivo acumulado
    masterfile/targeted_sampling/observations-cuad.csv e regenera summary-cuad.csv.
    Nao duplica: se o run_id ja estiver no acumulado, nao reescreve.
    """
    from masterfile.extract import OBSERVATIONS_COLUMNS, PKG_DIR

    out_dir = os.path.join(PKG_DIR, TARGETED_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    obs_path = os.path.join(out_dir, "observations-cuad.csv")
    target_ids = {t["full_op_id"] for t in targets}

    run_obs_path = os.path.join(run_dir, "observations.csv")
    if not os.path.exists(run_obs_path):
        print(f"[sample-ops] AVISO: {run_obs_path} nao existe (extracao do masterfile falhou?).")
        return obs_path

    with open(run_obs_path, encoding="utf-8") as f:
        new_rows = [r for r in csv.DictReader(f) if r["full_op_id"] in target_ids]

    existing: list[dict] = []
    if os.path.exists(obs_path):
        with open(obs_path, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    new_run_ids = {r["run_id"] for r in new_rows} | {run_id}
    if any(r["run_id"] in new_run_ids for r in existing):
        print(f"[sample-ops] run {run_id} ja estava no acumulado; nada adicionado.")
    else:
        write_header = not os.path.exists(obs_path)
        with open(obs_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OBSERVATIONS_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        existing.extend(new_rows)
        print(f"[sample-ops] {len(new_rows)} observacoes adicionadas a {obs_path}")

    # resumo por operador alvo (todas as runs dirigidas acumuladas)
    summary_path = os.path.join(out_dir, "summary-cuad.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["full_op_id", "label", "samples_before", "targeted_total",
                         "targeted_valid", "targeted_valid_with_quality", "runs"])
        for t in targets:
            rows = [r for r in existing if r["full_op_id"] == t["full_op_id"]]
            valid = [r for r in rows if _is_valid(r)]
            with_q = [r for r in valid if r.get("quality") not in ("", None)]
            writer.writerow([t["full_op_id"], t.get("label", ""), t.get("samples_before", ""),
                             len(rows), len(valid), len(with_q),
                             ";".join(sorted({r["run_id"] for r in rows}))])
    print(f"[sample-ops] resumo: {summary_path}")
    return obs_path


def report(plan_stats, targets: list[dict], min_samples: int) -> int:
    """Imprime amostras validas por operador alvo NESTA run. Devolve no. de alvos abaixo do minimo."""
    counts = {t["full_op_id"]: [0, 0, 0] for t in targets}  # total, validas, validas c/ quality
    for _, by_full in plan_stats.operator_stats.items():
        for full_op_id, op_stats in by_full.items():
            if full_op_id not in counts:
                continue
            for rec in op_stats.record_op_stats_lst:
                counts[full_op_id][0] += 1
                if not rec.failed_convert:
                    counts[full_op_id][1] += 1
                    if rec.quality is not None:
                        counts[full_op_id][2] += 1

    print("\n========== AMOSTRAS NESTA RUN (por operador alvo) ==========")
    print(f"{'full_op_id':24s} {'total':>5s} {'validas':>7s} {'c/quality':>9s}  label")
    below = 0
    for t in targets:
        total, valid, with_q = counts[t["full_op_id"]]
        flag = "" if valid >= min_samples else "   <-- abaixo do minimo"
        below += valid < min_samples
        print(f"{t['full_op_id']:24s} {total:5d} {valid:7d} {with_q:9d}  {t.get('label', '')}{flag}")
    print("=============================================================\n")
    return below


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(description="Amostragem dirigida de operadores do CUAD (fase de sampling).")
    p.add_argument("--ops-file", default=os.path.join(HERE, "target_ops_cuad.json"),
                   help="JSON com a lista de operadores alvo (chave 'operators' -> [{'full_op_id': ...}])")
    p.add_argument("--num-records", type=int, default=5,
                   help="Contratos de treino por operador (= amostras por operador). Max 25.")
    p.add_argument("--seed", type=int, default=42, help="Mesmo seed das runs originais (define os contratos).")
    p.add_argument("--exp-name", default=None, help="Default: cuad-targeted-all-n<N>-seed<seed>")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="Endpoint do scheduler.py")
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--dry-run", action="store_true",
                   help="So monta o plano e confere os ids alvo. Nao chama LLM nem grava no masterfile.")
    return p.parse_args()


def main() -> int:
    args = parse_arguments()
    if not 1 <= args.num_records <= 25:
        raise SystemExit("--num-records deve estar entre 1 e 25 (o validator conhece 25 contratos de treino).")

    with open(args.ops_file, encoding="utf-8") as f:
        ops_spec = json.load(f)
    targets = ops_spec["operators"]
    target_ids = {t["full_op_id"] for t in targets}
    min_samples = int(ops_spec.get("min_samples", args.num_records))
    exp_name = args.exp_name or f"cuad-targeted-all-n{args.num_records}-seed{args.seed}"
    print(f"[sample-ops] {len(target_ids)} operadores alvo; {args.num_records} contratos de treino; exp={exp_name}")

    cuad = load_cuad_demo()
    import palimpzest as pz
    from palimpzest.core.models import ExecutionStats
    from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

    policy = pz.MaxQuality()
    config = pz.QueryProcessorConfig(
        available_models=list(AVAILABLE_MODELS),
        api_base=args.api_base,
        policy=policy,
        verbose=False,
        optimizer_strategy="pareto",
        sentinel_execution_strategy="all",
        execution_strategy="parallel",
        max_workers=args.max_workers,
        allow_bonded_query=True,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=True,
        k=6,
        j=4,
        sample_budget=len(target_ids) * args.num_records,  # so' alimenta a barra de progresso
        seed=args.seed,
        exp_name=exp_name,
        priors=None,
    )

    # mesmos objetos do cuad-demo1.py
    validator = cuad.CUADValidator(num_contracts=25, seed=args.seed)
    test_dataset = cuad.CUADDataset(split="test", num_contracts=100, seed=args.seed)
    train = cuad.CUADDataset(split="train", num_contracts=args.num_records, seed=args.seed)
    train_dataset = {train.id: train}
    query = cuad.build_cuad_query(test_dataset, "one-convert")

    processor = QueryProcessorFactory.create_processor(query, config, train_dataset, validator)
    plan = processor._create_sentinel_plan(train_dataset)
    filter_sentinel_plan(plan, target_ids)

    if args.dry_run:
        print("[sample-ops] DRY-RUN OK: todos os operadores alvo existem no sentinel plan. Nada foi executado.")
        print(plan)
        return 0

    # masterfile: reserva runs/<run_id>/ e grava run_config.json
    from masterfile.hooks import MasterfileRun
    mf = MasterfileRun.start(
        exp_name=exp_name,
        benchmark="cuad",
        args=args,
        pz_config=config,
        policy=policy,
        extra={
            "sentinel_execution_strategy": "all",
            "sentinel_strategy": "ALL",
            "targeted_sampling": True,
            "targeted_ops": sorted(target_ids),
            "train_split_size": args.num_records,
            "validator_num_contracts": 25,
            "num_fields": cuad.NUM_FIELDS_TO_EXTRACT_PER_CONTRACT,
            "iou_thresh": cuad.IOU_THRESH,
        },
    )

    StrategyCls = make_strategy_class()
    strategy = StrategyCls(target_ids, **config.to_dict())

    stats = ExecutionStats(execution_id=processor.execution_id())
    stats.start()
    plan_stats = strategy.execute_sentinel_plan(plan, train_dataset, validator)
    stats.add_plan_stats(plan_stats)
    stats.finish_optimization()
    stats.finish()
    stats_dict = stats.to_json()

    os.makedirs("opt-profiling-data", exist_ok=True)
    with open(f"opt-profiling-data/{exp_name}-stats.json", "w") as f:
        json.dump(stats_dict, f)

    mf.finish(stats_dict, metrics=None)
    try:
        append_targeted_observations(mf.run_dir, mf.run_id, targets)
    except Exception as e:  # noqa: BLE001 — nao perder a run por causa do acumulado
        print(f"[sample-ops] AVISO: falha ao atualizar o acumulado ({e!r}). "
              f"Os dados estao em {mf.run_dir}/observations.csv")

    below = report(plan_stats, targets, min_samples)
    print(f"Optimization (sampling) time: {stats.optimization_time:.1f}s | cost: {stats.optimization_cost}")
    if below:
        print(f"[sample-ops] {below} operador(es) com menos de {min_samples} amostras validas nesta run. "
              f"Rode de novo (ex.: --num-records {args.num_records + 1}) ou com um --ops-file so com eles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
