"""
masterfile.hooks — integração do masterfile com os scripts de benchmark.

Substitui os blocos inline (~40 linhas cada) que estavam duplicados no
cuad-demo1.py e nos scripts do BioDEX. Cada script passa a precisar de só duas
chamadas:

    from masterfile.hooks import MasterfileRun

    # 1) ANTES de optimize_and_run (depois de exp_name, policy e config existirem)
    mf = MasterfileRun.start(
        exp_name=exp_name,
        benchmark="biodex",              # chave de masterfile/benchmarks.py
        args=args,                       # argparse.Namespace -> todos os parâmetros
        pz_config=config,                # pz.QueryProcessorConfig
        policy=policy,                   # opcional; se None, usa config.policy
        extra={"test_split_size": 250},  # qualquer metadado específico do benchmark
    )

    # 2) DEPOIS de calcular as métricas
    mf.finish(execution_stats_dict, metrics=stats_dict)

O QUE CADA ETAPA GRAVA em masterfile/runs/<run_id>/
---------------------------------------------------
start():   run_config.json            configuração completa da run
           prior_store_snapshot.json  cópia da memória do CXTBAY (se existir)
finish():  run.json, observations.csv, final_exec.csv, operators.csv, verify.txt
           (via extract.extract_run)

GARANTIAS
---------
- start() PODE falhar (import, disco): falha cedo, antes de gastar GPU.
- finish() NUNCA levanta exceção: uma falha no masterfile não derruba uma run
  que já custou horas de GPU. Os arquivos brutos do benchmark (stats/profiling
  e metrics.json) continuam salvos e podem ser reprocessados pelo CLI do
  extract.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any

from .benchmarks import get_spec
from .extract import allocate_run_dir, extract_run

DEFAULT_PRIOR_PATH = os.path.join("operator_memory", "operator_priors.json")


def _prior_store_info(prior_path: str, run_dir: str) -> dict[str, Any]:
    """Hash e resumo da memória do CXTBAY no início da run (+ snapshot em disco)."""
    if not os.path.exists(prior_path):
        return {
            "prior_store_hash": None,
            "prior_store_num_ops": 0,
            "prior_store_training_runs": 0,
        }
    with open(prior_path, "rb") as f:
        raw = f.read()
    mem = json.loads(raw.decode("utf-8"))
    shutil.copy(prior_path, os.path.join(run_dir, "prior_store_snapshot.json"))
    return {
        "prior_store_hash": hashlib.sha256(raw).hexdigest()[:16],
        "prior_store_num_ops": len(mem.get("operators", {})),
        "prior_store_training_runs": mem.get("meta", {}).get("training_runs"),
    }


def build_run_config(exp_name: str,
                     benchmark: str,
                     args: Any = None,
                     pz_config: Any = None,
                     policy: Any = None,
                     extra: dict | None = None) -> dict[str, Any]:
    """
    Monta o dicionário do run_config.json. Separado de start() para poder ser
    testado sem tocar em disco.

    Ordem de precedência (a última vence): args -> campos derivados -> extra.
    """
    cfg: dict[str, Any] = dict(vars(args)) if args is not None else {}

    policy = policy if policy is not None else getattr(pz_config, "policy", None)
    spec = get_spec(benchmark)

    cfg.update({
        "benchmark": benchmark,
        "exp_name": exp_name,
        "policy_class": type(policy).__name__ if policy is not None else None,
        "policy_str": str(policy) if policy is not None else None,
        "available_models": [str(m) for m in (getattr(pz_config, "available_models", None) or [])],
        "optimizer_strategy_effective": getattr(pz_config, "optimizer_strategy", None),
        "max_workers_effective": getattr(pz_config, "max_workers", None),
        "quality_metric": spec.quality_metric if spec else None,
        "num_semantic_ops": spec.num_semantic_ops if spec else None,
        "config_source": "sidecar",
        "campaign": os.environ.get("MASTERFILE_CAMPAIGN", "behavior"),
        "hardware_note": os.environ.get("MASTERFILE_HARDWARE", ""),
        "models_loaded": os.environ.get("MASTERFILE_MODELS_LOADED", ""),
    })

    # Nome canônico usado pelo extract.verify() e pelo CLI (parse_config_from_exp_name)
    strategy = cfg.get("sentinel_execution_strategy")
    if strategy:
        cfg["sentinel_strategy"] = str(strategy).upper()

    cfg.update(extra or {})
    return cfg


class MasterfileRun:
    """Uma run do masterfile: diretório reservado + configuração congelada."""

    def __init__(self, run_dir: str, run_id: str, config: dict[str, Any]):
        self.run_dir = run_dir
        self.run_id = run_id
        self.config = config

    @classmethod
    def start(cls,
              exp_name: str,
              benchmark: str,
              args: Any = None,
              pz_config: Any = None,
              policy: Any = None,
              extra: dict | None = None,
              prior_path: str = DEFAULT_PRIOR_PATH) -> "MasterfileRun":
        """Reserva masterfile/runs/<run_id>/ e grava run_config.json."""
        run_dir, run_id = allocate_run_dir(exp_name)
        config = build_run_config(exp_name, benchmark, args, pz_config, policy, extra)
        config.update(_prior_store_info(prior_path, run_dir))

        with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        print(f"[masterfile] run reservada: {run_id} ({benchmark})")
        return cls(run_dir, run_id, config)

    def finish(self, execution_stats_dict: dict, metrics: dict | None = None) -> dict | None:
        """Extrai os CSVs e o run.json. Nunca levanta exceção; devolve o run.json ou None."""
        try:
            return extract_run(
                execution_stats_dict,
                self.run_dir,
                self.run_id,
                config=self.config,
                metrics=metrics,
            )
        except Exception as e:  # noqa: BLE001 — proteger a run é o objetivo
            print(f"[masterfile] AVISO: extracao falhou ({e!r}). Os arquivos brutos do "
                  f"benchmark foram preservados; reprocesse com: python masterfile/extract.py "
                  f"--stats <arquivo -stats.json ou -profiling.json> --config "
                  f"{os.path.join(self.run_dir, 'run_config.json')} --verify")
            return None