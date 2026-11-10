# Tutorial: amostragem dirigida de operadores (CUAD)

Garante que uma lista fixa de operadores físicos tenha pelo menos 5 amostras na fase de *operator sampling*. Essas amostras alimentam os boxplots de qualidade, custo e latência.

O script **não altera** nenhuma lógica do Abacus nem arquivo existente. Ele só adiciona arquivos novos.

---

## 1. Arquivos novos

| Arquivo | Onde fica | Papel |
|---|---|---|
| `cuad-sample-ops.py` | `palimpzest/abacus-research/` | Script de amostragem dirigida |
| `target_ops_cuad.json` | `palimpzest/abacus-research/` | Lista dos operadores alvo (13 por padrão) |
| `TUTORIAL_AMOSTRAGEM_DIRIGIDA.md` | `palimpzest/abacus-research/` | Este tutorial |
| `cuad_sample_ops_axia.sh` | `POLYAXON_KUNUMI/` | Sobe vLLM e scheduler e roda o script (equivale ao `cuad_tests_axia.sh`) |
| `polyaxon_job_cuad_sample_ops.yaml` | `POLYAXON_KUNUMI/` | Job Polyaxon (equivale ao `polyaxon_job_cuad.yaml`) |

---

## 2. O que o script faz

1. Monta **a mesma query e a mesma configuração** do `cuad-demo1.py`:
   - mesmos 3 modelos (Llama-3.1-8B, Llama-3.3-70B e GPT-OSS-120B, sem Qwen);
   - mesmo `api_base` (`http://localhost:8000/v1`, o `scheduler.py`);
   - mesmas flags `allow_*` e `seed=42`.

   Isso é obrigatório. O `full_op_id` é um hash dos parâmetros do operador, e com outra configuração os ids não batem.
2. Gera o *sentinel plan* normal do Palimpzest e **mantém só os operadores alvo**. O scan também fica.
3. Executa a estratégia que já existe no Palimpzest, `AllSamplingExecutionStrategy` (`sentinel_execution_strategy="all"`). Cada operador alvo roda em cada um dos N contratos de treino.
   - A execução, o validator (F1 do CUAD) e o registro em `SentinelPlanStats` são os mesmos do MAB e do CXTBAY, então as medidas são comparáveis.
   - Os N contratos são os N primeiros do mesmo embaralhamento (`seed=42`) usado nas runs normais.
4. **Não** executa o plano final nos 100 contratos de teste. Só a fase de sampling interessa aqui.

### Modelos: vLLM e scheduler

Na execução real, o script chama os modelos **da mesma forma que o `cuad-demo1.py`**:

```
cuad-sample-ops.py ──> scheduler.py (:8000) ──> vLLM Llama-3.1-8B   (:8005)
                                            ├─> vLLM Llama-3.3-70B  (:8006)
                                            └─> vLLM GPT-OSS-120B   (:8008)
RAGConvert ──> embedding nomic-embed-text-v1 (:8001)
```

`vllm-axia.sh` sobe os servidores vLLM e `scheduler.py` faz o roteamento na porta 8000. Os dois já existem e **não foram alterados**. Os operadores alvo usam só esses 3 modelos, mais o embedding no caso dos RAG, então nada precisa mudar no vLLM.

> Se um dia você incluir operadores com Qwen3-235B no `target_ops_cuad.json`, precisa descomentar o Qwen no `vllm-axia.sh`, no `scheduler.py` **e** em `AVAILABLE_MODELS` no `cuad-sample-ops.py`. Sem isso, o dry-run acusa que os ids não existem.

---

## 3. Saídas (dentro de `abacus-research/masterfile/`)

| Caminho | Conteúdo |
|---|---|
| `masterfile/targeted_sampling/observations-cuad.csv` | **Arquivo acumulado** só com as amostragens dirigidas: todas as runs deste script, sem o scan. Tem as mesmas colunas do `observations.csv` normal. Cada run é adicionada uma única vez. |
| `masterfile/targeted_sampling/summary-cuad.csv` | Por operador alvo: `samples_before` (da tabela), `targeted_total`, `targeted_valid` (sem `failed_convert`), `targeted_valid_with_quality` e as runs. É regenerado a cada execução. |
| `masterfile/runs/cuad-targeted-all-n5-seed42[__00X]/` | Run completa no formato padrão: `run_config.json`, `run.json`, `observations.csv`, `operators.csv`, `final_exec.csv` (vazio) e `verify.txt`. |
| `opt-profiling-data/cuad-targeted-all-n5-seed42-stats.json` | `ExecutionStats` bruto, igual ao do `cuad-demo1.py`. |

Na Axia, o `cuad_sample_ops_axia.sh` também copia `targeted_sampling/` e `runs/cuad-targeted-*` para `uploads/masterfile_targeted/`. Assim eles aparecem nos *artifacts* do Polyaxon para download.

**Observação:** o `masterfile/consolidate.py` também lê as runs `cuad-targeted-*` ao regenerar o `masterfile-cuad.csv`. Para separar as runs dirigidas das normais, filtre por `sentinel_strategy == "ALL"`, `campaign == "targeted-fill"` ou pelo prefixo do `run_id`.

---

## 4. Como rodar na Axia (Polyaxon)

### 4.1 Subir os arquivos
1. Copie `cuad-sample-ops.py` e `target_ops_cuad.json` para `palimpzest/abacus-research/` na base de código da Axia, no mesmo lugar do `cuad-demo1.py`.
   - **Se a imagem `otimizacao-abacus` for reconstruída** com esses arquivos, eles já vão para `/app` e de lá para o scratch.
   - **Se não for reconstruída**, envie os dois junto com o upload do job, na mesma pasta do `.sh`. O YAML copia os dois para `abacus-research/` no scratch.
2. Coloque `cuad_sample_ops_axia.sh` e `polyaxon_job_cuad_sample_ops.yaml` junto dos seus outros scripts (`POLYAXON_KUNUMI/`).

### 4.2 Disparar o job
Use o mesmo comando que você usa para o `polyaxon_job_cuad.yaml`, trocando o arquivo. Por exemplo:

```bash
polyaxon run -f polyaxon_job_cuad_sample_ops.yaml -u
```

O `-u` envia a pasta atual para `uploads`, que é o `workingDir` do job.

### 4.3 O que acontece no job
1. Setup: apaga `operator_memory`, copia `/app` para o scratch e, se vierem no upload, copia os dois arquivos.
2. **Dry-run (sem GPU e sem LLM):** confere se os 13 ids existem no sentinel plan. Se não existirem, o job **para antes de subir o vLLM** e não gasta os ~20 min de espera.
3. Sobe o `vllm-axia.sh` e espera 1200 s. Depois sobe o `scheduler.py` e espera 60 s.
4. Roda `cuad-sample-ops.py --num-records 5`.
5. Copia os resultados para `uploads/masterfile_targeted/`.

Parâmetros por variável de ambiente, se quiser mudar sem editar o `.sh`: `NUM_RECORDS` (padrão 5), `OPS_FILE` (padrão `target_ops_cuad.json`) e `SEED` (padrão 42).

### 4.4 Conferir o resultado
No final do log (`var/logs/cuad-sample-ops.log`) aparece uma tabela parecida com esta:

```
========== AMOSTRAS NESTA RUN (por operador alvo) ==========
full_op_id               total validas c/quality  label
81dd1f8073-dc91ce0947        5       5         5  RAG[GPT-OSS120B;chunk=4000;nchunks=2]
...
```

Um operador com `<-- abaixo do minimo` teve `failed_convert` (timeout, JSON inválido etc.). Nesse caso, veja a seção 6.

---

## 5. Como rodar localmente (sem Polyaxon)

```bash
cd palimpzest/abacus-research

# 1) só conferir os ids (não precisa de vLLM nem de scheduler)
python cuad-sample-ops.py --dry-run

# 2) com os modelos no ar (vllm + scheduler na porta 8000)
bash ./vllm-axia.sh > var/logs/vllm.log 2>&1 &     # ou o seu vllm.sh local
python scheduler.py > var/logs/scheduler.log 2>&1 &
MASTERFILE_CAMPAIGN=targeted-fill python cuad-sample-ops.py --num-records 5
```

### Opções do script

| Opção | Padrão | Descrição |
|---|---|---|
| `--ops-file` | `target_ops_cuad.json` | Lista de operadores alvo |
| `--num-records` | 5 | Contratos de treino, igual ao número de amostras por operador (de 1 a 25) |
| `--seed` | 42 | Mesmo seed das runs originais |
| `--exp-name` | `cuad-targeted-all-n<N>-seed<seed>` | Nome da run no masterfile. Se já existir, vira `__002`, `__003`... e nada é sobrescrito |
| `--api-base` | `http://localhost:8000/v1` | Endpoint do scheduler |
| `--max-workers` | 64 | Paralelismo, igual ao `cuad-demo1.py` |
| `--dry-run` | — | Só monta o plano e confere os ids |

---

## 6. Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Dry-run: `operador(es) alvo nao existem no sentinel plan` | A configuração diverge das runs originais: modelos, flags ou versão do código | Compare `AVAILABLE_MODELS` do script com o `cuad-demo1.py`. O log lista os ids disponíveis |
| Operador com menos de 5 válidas | `failed_convert` em algum contrato | Rode de novo com um `--ops-file` só com os faltantes (copie o JSON e deixe só eles) ou com `--num-records 6`. O acumulado soma as runs, e o `summary-cuad.csv` mostra o total |
| `quality` vazia | O validator não pontuou | Confira o log. Nas runs normais recentes (`pos-fix-quality`) a quality é preenchida |
| Job morre esperando o vLLM | Problema no vLLM ou nas GPUs | Veja `var/logs/vllm.log` e `saida_VLLM_*.txt` |
| RAG falha | Servidor de embedding (:8001) fora do ar | Veja `var/logs/embedding.log` |

---

## 7. Mudar o alvo

Edite `target_ops_cuad.json`. Só a chave `full_op_id` é obrigatória:

```json
{
  "min_samples": 5,
  "operators": [
    {"full_op_id": "81dd1f8073-dc91ce0947", "label": "RAG[GPT-OSS120B;chunk=4000;nchunks=2]", "samples_before": 0}
  ]
}
```

Os ids vêm do `operators.csv` ou do `observations.csv` do masterfile, na coluna `full_op_id`.
