# masterfile

Registro estruturado de cada run de benchmark (CUAD, BioDEX, ...) em CSVs estáveis,
para análise entre runs, estratégias e benchmarks.

## Arquivos

| Arquivo | Papel |
|---|---|
| `extract.py` | Converte o `ExecutionStats` do Palimpzest em `run.json` + CSVs. Também é o CLI para reprocessar runs antigas. |
| `benchmarks.py` | Única fonte da verdade sobre as métricas de cada benchmark (qual chave é a qualidade, quantos operadores o plano tem). |
| `hooks.py` | API usada pelos scripts de benchmark: `MasterfileRun.start()` e `.finish()`. |
| `consolidate.py` | Varre `runs/*/` e regenera os masterfiles de um benchmark em `out/` (idempotente). |

## Uso dentro de um script de benchmark

```python
from masterfile.hooks import MasterfileRun

# antes de optimize_and_run (exp_name, policy e config já definidos)
mf = MasterfileRun.start(exp_name=exp_name, benchmark="biodex", args=args,
                         pz_config=config, extra={"test_split_size": 250})

# depois de calcular as métricas
mf.finish(execution_stats_dict, metrics=stats_dict)
```

`start()` pode falhar (falha cedo, antes de gastar GPU). `finish()` nunca levanta exceção.

## O que cada run gera em `masterfile/runs/<run_id>/`

| Arquivo | Conteúdo |
|---|---|
| `run_config.json` | todos os argumentos + política, modelos, otimizador, benchmark, snapshot da memória do CXTBAY |
| `prior_store_snapshot.json` | cópia de `operator_memory/operator_priors.json` no início da run (se existir) |
| `run.json` | config + tempos/custos + **`quality`, `quality_metric`** + métricas específicas do benchmark |
| `observations.csv` | 1 linha por (operador × registro) na amostragem |
| `final_exec.csv` | 1 linha por (operador × registro) na execução final |
| `operators.csv` | 1 linha por operador físico visto na run |
| `verify.txt` | alertas de consistência (custo, budget, identidade, cobertura) |

`<run_id>` é o `exp_name`; se já existir, vira `<exp_name>__002`, `__003`... Nada é sobrescrito.

## Qualidade entre benchmarks

| Benchmark | `quality_metric` | Outras métricas copiadas para o `run.json` | Operadores |
|---|---|---|---|
| cuad | `f1` | `precision`, `recall`, `f1` | 1 |
| biodex | `rp@5` | `rp@5`, `bad`/`failed`, `n_expected`, `n_returned`, `n_missing`, `empty_reaction_labels_frac`, `silent_error_files` | 3 |

Para comparar benchmarks, use sempre a coluna `quality`. Para adicionar um benchmark,
acrescente uma entrada em `BENCHMARKS` (`benchmarks.py`).

## Consolidar (gerar os masterfiles)

```bash
python masterfile/consolidate.py --benchmark cuad
python masterfile/consolidate.py --benchmark biodex
```

Gera em `masterfile/out/`: `masterfile-<bench>.csv` (observações), `-final.csv`,
`-runs.csv` (1 linha por run, com `quality`) e `-operators.csv`. Reconstrói do zero a cada
execução. Runs antigas (schema v1) são normalizadas na leitura, sem reprocessar:
`sentinel_strategy` vem de `sentinel_execution_strategy` e `quality` vem do `f1`.
Runs sem benchmark identificável são ignoradas com aviso.

No BioDEX, filtre as três variantes pela coluna `benchmark_variant`
(`maxquality-greedy`, `maxquality-at-cost`, `pareto-cascades-ablation`).

## Reprocessar uma run antiga (CLI)

```bash
# CUAD
python masterfile/extract.py --stats opt-profiling-data/<exp>-stats.json --verify
# BioDEX (os scripts salvam -profiling.json)
python masterfile/extract.py --stats max-quality-at-cost-data/<exp>-profiling.json --benchmark biodex --verify
# com o run_config.json de uma run ao vivo (config completa em vez de inferida do nome)
python masterfile/extract.py --stats <...> --config masterfile/runs/<run_id>/run_config.json --verify
```

Sem `--config`, a configuração é inferida do nome do experimento
(`-k6-j4-budget150-seed42`, estratégia e benchmark pelo prefixo).

## Histórico do schema

- **v1**: run.json com `precision`/`recall`/`f1` (só CUAD).
- **v2**: `quality`, `quality_metric`, métricas por benchmark; `verify()` considera
  `k*j*num_semantic_ops`; CLI aceita `-profiling.json`. CSVs inalterados.
  `consolidate.py` normaliza runs v1 na leitura e não mistura mais runs sem benchmark.