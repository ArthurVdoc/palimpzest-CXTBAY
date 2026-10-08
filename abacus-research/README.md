## Tutorial rápido: subir vllm, scheduler e rodar o biodex-demo

Este README descreve os passos para iniciar o ambiente `vllm`, executar o `scheduler.py` e depois o `biodex-demo.py` no repositório `abacus-research`.

### Avisos importantes (leia antes de começar)
- ALERTA: sempre ative o ambiente Conda indicado antes de qualquer comando:

  ```bash
  conda activate /scratch/global/abacus
  ```

- ALERTA: verifique se a GPU está disponível e não ocupada. Execute `nvidia-smi` e confirme se há memória/uso livre antes de prosseguir:

  ```bash
  nvidia-smi
  ```

  Se a GPU estiver em uso por outro processo, finalize ou escolha outra máquina/slot antes de prosseguir.

### Rodando com o TMUX vários testes automatizados

```bash
tmux new -s biodex-run
cd /scratch/global/palimpzest/abacus-research/
./run-biodex-multiple-budget.sh
```

```bash
tmux new -s cuad-run
cd /scratch/global/palimpzest/abacus-research/
./run-cuad-multiple-budget.sh
```

para acompanhar:
```bash
tmux attach -t biodex-run
tmux attach -t biodex-tests
```

e para matar os tmux

```bash
tmux kill-session -t biodex-run
tmux kill-session -t biodex-tests
```

comandos (dentro do tmux):
CTRL+b: ativar o modo de comandos
-> w: ver as janelas na sessão (depois de apertar ctrl+b, aperta w)

### Passo a passo

1) Ativar o ambiente Conda e entrar no diretório correto

```bash
source /opt/miniconda/bin/activate
conda activate /home/luizaregi/.conda/envs/otimizacao-abacus
cd /home/luizaregi/palimpzest/backup_morpheu/palimpzest/abacus-research
```

2) Subir o vllm

No primeiro terminal (ainda no diretório `abacus-research`):

```bash
./vllm.sh
```

3) Conferir modelos e portas

É obrigatório que os nomes dos modelos e as portas em `vllm.sh` sejam exatamente os mesmos que aparecem em `scheduler.py` e em `biodex-demo.py`.


4) Iniciar o `scheduler.py`

Abra um segundo terminal, ative o ambiente e execute:

```bash
source /opt/miniconda/bin/activate
conda activate /home/luizaregi/.conda/envs/otimizacao-abacus
cd /home/luizaregi/palimpzest/backup_morpheu/palimpzest/abacus-research
python scheduler.py
```

5) Aguarde todos os modelos subirem

No terminal onde `./vllm.sh` está rodando, aguarde até ver logs que confirmem que cada modelo está carregado e a porta está aberta.

6) Rodar o `biodex-demo.py`

Quando todos os modelos estiverem prontos, abra um terceiro terminal, ative o ambiente e execute:

```bash
source /opt/miniconda/bin/activate
conda activate /home/luizaregi/.conda/envs/otimizacao-abacus
cd /home/luizaregi/palimpzest/backup_morpheu/palimpzest/abacus-research
python biodex-demo.py --k 3 --j 3 --progress --sentinel-execution-strategy mab
#ou
python cuad-demo1.py --k 6 --j 4 --sample-budget 15 --sentinel-execution-strategy bay
# Executa com algoritmo bayesiano contextual
python cuad-demo1.py --k 6 --j 4 --sample-budget 30  --sentinel-execution-strategy cxtbay 
```
opções para sentinel-execution-strategy: MAB/ALL/BAY

### Verificações e troubleshooting rápido
- Se `biodex-demo.py` não conectar a um modelo, confira:
  - As portas listadas em `vllm.sh` correspondem às usadas por `scheduler.py` e `biodex-demo.py`.
  - Os modelos foram carregados corretamente no log do `vllm.sh`.
  - `nvidia-smi` mostra GPU com memória livre suficiente.

- Logs úteis:
  - Saída de `./vllm.sh` (logs de cada servidor de modelo)
  - Saída de `python scheduler.py` (erros de conexão/tempo limite)

### Dicas
- Use três terminais separados (vllm, scheduler, demo) para facilitar inspeção dos logs.
- Se precisar checar os detalhes dentro dos scripts, procure por strings que definem portas e nomes de modelo — normalmente há variáveis ou argumentos no início dos scripts.

### Resumo rápido (comandos)

```bash
# Terminal 1
conda activate /scratch/global/abacus
cd /scratch/global/palimpzest/abacus-research/
./vllm.sh

# Terminal 2
conda activate /scratch/global/abacus
cd /scratch/global/palimpzest/abacus-research/
python scheduler.py

# Terminal 3 (após modelos estarem prontos)
conda activate /scratch/global/abacus
cd /scratch/global/palimpzest/abacus-research/
python biodex-demo.py --k 3 --j 3 --progress
python cuad-demo1.py --k 3 --j 3 --progress
ps -fp 
```
