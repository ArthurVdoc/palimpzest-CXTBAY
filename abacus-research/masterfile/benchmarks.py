"""
masterfile.benchmarks — especificação das métricas de cada benchmark.

POR QUE ESTE MÓDULO EXISTE
--------------------------
Cada benchmark mede qualidade de um jeito diferente:

    CUAD   -> F1 (com precision e recall)
    BioDEX -> RP@5 (rank-precision at 5)

O extrator (extract.py) e o run.json precisam de uma forma ÚNICA de ler a
qualidade, para que runs de benchmarks diferentes possam ser consolidadas na
mesma tabela. Este módulo é o único lugar que sabe "qual chave do metrics.json
é a qualidade de qual benchmark".

COMO ADICIONAR UM BENCHMARK
---------------------------
Acrescente uma entrada em BENCHMARKS. Nada mais precisa mudar:

    "mmqa": BenchmarkSpec(
        name="mmqa",
        quality_metric="f1",
        metric_keys=("f1",),
        num_semantic_ops=5,
    ),

SAÍDA PADRONIZADA (o que vai para o run.json)
---------------------------------------------
    quality          valor da métrica de qualidade do benchmark
    quality_metric   nome dessa métrica ("f1", "rp@5", ...)
    <metric_key>     cada chave de metric_keys presente no metrics.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    # chave do metrics.json que é a qualidade final do plano
    quality_metric: str
    # chaves do metrics.json copiadas para o run.json (além de quality)
    metric_keys: tuple[str, ...]
    # nº de operadores semânticos do plano lógico; usado pelo verify() para
    # checar se o budget cobre ao menos a primeira rodada (k * j por operador)
    num_semantic_ops: int


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "cuad": BenchmarkSpec(
        name="cuad",
        quality_metric="f1",
        metric_keys=("precision", "recall", "f1"),
        num_semantic_ops=1,          # 1 sem_map com 41 campos
    ),
    "biodex": BenchmarkSpec(
        name="biodex",
        quality_metric="rp@5",
        metric_keys=(
            "rp@5",
            "bad",                          # biodex-demo: registros com erro no cálculo do RP@5
            "failed",                       # max-quality-at-cost / pareto-cascades: idem
            "n_expected",                   # tamanho do split de teste
            "n_returned",                   # registros que voltaram da execução
            "n_missing",                    # pmids sem saída (contam como perda de qualidade)
            "empty_reaction_labels_frac",   # retrieve vazio: sinal de falha de embeddings
            "silent_error_files",           # arquivos novos em *-errors/ durante a run
        ),
        num_semantic_ops=3,          # sem_map -> retrieve -> sem_map
    ),
}


def get_spec(benchmark: str | None) -> BenchmarkSpec | None:
    """Devolve a especificação do benchmark, ou None se desconhecido."""
    if not benchmark:
        return None
    return BENCHMARKS.get(str(benchmark).lower())


def normalize_metrics(benchmark: str | None, metrics: dict[str, Any] | None) -> dict[str, Any]:
    """
    Converte o metrics.json de um benchmark no bloco padronizado do run.json.

    - Benchmark conhecido: quality = metrics[spec.quality_metric], mais as
      metric_keys presentes.
    - Benchmark desconhecido: usa metrics["quality"] / metrics["quality_metric"]
      se o script do benchmark já os tiver gravado; senão, quality = None.

    Nunca levanta exceção: chaves ausentes simplesmente não aparecem.
    """
    metrics = metrics or {}
    spec = get_spec(benchmark)
    out: dict[str, Any] = {}

    if spec is not None:
        out["quality_metric"] = spec.quality_metric
        out["quality"] = metrics.get(spec.quality_metric)
        for key in spec.metric_keys:
            if key in metrics:
                out[key] = metrics[key]
    else:
        out["quality_metric"] = metrics.get("quality_metric")
        out["quality"] = metrics.get("quality")

    return out