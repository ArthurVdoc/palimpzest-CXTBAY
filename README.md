# CXTBAY: amostragem Bayesiana com persistência de contexto para operadores semânticos

**"Ajuste Automático de Parâmetros Bayesiano com Persistência de Contexto para Operadores Semânticos"**

**Autores:** Arthur Vinícius, Luiza Regi, Rafael Alexandre, Willian Barreiros Jr. e George Teodoro
Departamento de Ciência da Computação, Universidade Federal de Minas Gerais (UFMG)

Este repositório é um fork do [Palimpzest](https://github.com/mitdbg/palimpzest) que estende o otimizador [Abacus](https://github.com/mitdbg/palimpzest/tree/main/abacus-research) com duas novas estratégias de amostragem de operadores físicos.

## O que este trabalho adiciona

O Abacus escolhe quais operadores físicos avaliar tratando a busca como um problema de *Multi-Armed Bandit* (MAB) guiado por intervalos de confiança. Essa abordagem recomeça do zero a cada otimização: nada do que foi aprendido em uma execução é aproveitado na seguinte.

Este repositório acrescenta:

| Estratégia | Flag | Ideia |
|---|---|---|
| **MAB** | `mab` | Baseline original do Abacus (UCB/LCB). |
| **BAY** | `bay` | Substitui os intervalos de confiança por distribuições a posteriori (Beta para qualidade, Gaussiana para custo e tempo) atualizadas com *Thompson Sampling*. A probabilidade de um operador pertencer à fronteira de Pareto é estimada por simulação de Monte Carlo. |
| **CXTBAY** | `cxtbay` | BAY com memória: as distribuições a posteriori são persistidas entre execuções. Para operadores nunca amostrados, um Processo Gaussiano (kernel RBF) sobre *embeddings* de operadores interpola priors informativas a partir dos operadores mais parecidos. |

## Organização do repositório

```
src/palimpzest/                  Palimpzest com as modificações deste trabalho
  query/execution/               Estratégias de execução (MAB, BAY, CXTBAY)
abacus-research/                 Scripts de experimento
  cuad-demo*.py                  Benchmark CUAD
  biodex-*.py                    Benchmark BioDEX
  mmqa-demo.py                   Benchmark MMQA
  vllm*.sh, start_vllm_servers.sh  Subida dos servidores vLLM
  scheduler*.py, server_proxy.py Roteamento das requisições entre os modelos
  operator_memory/               Memória persistida do CXTBAY (posteriors por operador)
  *-priors*.json                 Priors usadas nos experimentos
README_PALIMPZEST.md             README original do Palimpzest
```

## Como reproduzir

### 1. Instalação

```bash
git clone https://github.com/ArthurVdoc/palimpzest-CXTBAY.git
cd palimpzest-CXTBAY
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install vllm
```

### 2. Dados

Os conjuntos de dados não são distribuídos neste repositório. Baixe-os das fontes originais e coloque dentro de `abacus-research/`:

- **CUAD** ([Atticus Project](https://www.atticusprojectai.org/cuad)): em `abacus-research/cuad-data/`
- **BioDEX** ([repositório oficial](https://github.com/KarelDO/BioDEX)): os índices são gerados com `helper-scripts/biodex-gen-index.py`

### 3. Modelos

Os experimentos usam quatro modelos servidos localmente com [vLLM](https://github.com/vllm-project/vllm):

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen3-235B-A22B`
- `openai/gpt-oss-120b`

### 4. Execução

São três processos, cada um em seu terminal, todos a partir de `abacus-research/`.

```bash
# Terminal 1: servidores de modelo
./vllm.sh

# Terminal 2: scheduler (aguarde os modelos terminarem de carregar)
python scheduler.py

# Terminal 3: experimento
python cuad-demo1.py --k 6 --j 4 --sample-budget 40 --sentinel-execution-strategy mab
python cuad-demo1.py --k 6 --j 4 --sample-budget 40 --sentinel-execution-strategy bay
python cuad-demo1.py --k 6 --j 4 --sample-budget 40 --sentinel-execution-strategy cxtbay
```

Os nomes dos modelos e as portas precisam ser os mesmos em `vllm.sh`, em `scheduler.py` e no script do experimento.

Use `python cuad-demo1.py --help` para a lista completa, incluindo a política de otimização e o limite de custo.

### 5. Memória do CXTBAY

O CXTBAY lê e atualiza a memória em `abacus-research/operator_memory/`. Cada execução com `--sentinel-execution-strategy cxtbay` acumula conhecimento para as seguintes.

Para começar de uma memória vazia, mova ou apague o arquivo `operator_priors.json` dessa pasta.

## Como citar

```bibtex
@inproceedings{vinicius2026cxtbay,
  title     = {Ajuste Autom{\'a}tico de Par{\^a}metros Bayesiano com Persist{\^e}ncia de Contexto para Operadores Sem{\^a}nticos},
  author    = {Vin{\'i}cius, Arthur and Regi, Luiza and Alexandre, Rafael and Barreiros Jr., Willian and Teodoro, George},
  booktitle = {Anais do Simp{\'o}sio em Sistemas Computacionais de Alto Desempenho (SSCAD)},
  year      = {2026}
}
```

## Créditos e licença

Este trabalho é construído sobre o [Palimpzest](https://github.com/mitdbg/palimpzest) e o Abacus, do MIT Data Systems Group. O README original está preservado em [`README_PALIMPZEST.md`](README_PALIMPZEST.md).

O código é distribuído sob a licença MIT, a mesma do projeto original. Veja [`LICENSE`](LICENSE).

## Agradecimentos

Trabalho parcialmente financiado por CNPq, CAPES, FAPEMIG e Instituto Kunumi.
