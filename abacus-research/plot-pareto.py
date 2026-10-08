#!/usr/bin/env python3
"""
Percorre um diretório de arquivos JSON de espaço de busca (Abacus) e gera um
único gráfico quality x cost:

  - O search space completo (arquivo com mais operadores, tipicamente 121) é
    desenhado UMA vez, com cada operador como um ponto azul claro.
  - Os operadores que aparecem em QUALQUER fronteira de Pareto (arquivos com
    menos operadores, tipicamente 7) são destacados em vermelho. O casamento é
    feito pelo 'full_op_id', já que os arquivos de fronteira não trazem
    cost/quality (vêm como None) — os valores vêm sempre do search space.

Uso:
    python plot_pareto.py <diretorio_com_jsons>
"""

import os
import sys
import glob
import json

import matplotlib.pyplot as plt


def extrair_operadores(path):
    """Retorna dict full_op_id -> (cost, quality) para os operadores do arquivo.

    cost/quality podem ser None (caso dos arquivos de fronteira de Pareto).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ops = {}
    for group in data.get("groups", {}).values():
        for op in group.get("operators", []):
            pc = op.get("plan_cost") or {}
            ops[op["full_op_id"]] = (pc.get("cost"), pc.get("quality"))
    return ops


def main():
    if len(sys.argv) != 2:
        print("Uso: python plot_pareto.py <diretorio_com_jsons>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Erro: '{directory}' não é um diretório válido.")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not json_files:
        print(f"Nenhum arquivo .json encontrado em '{directory}'.")
        sys.exit(1)

    print(f"Encontrados {len(json_files)} arquivos .json. Processando...")

    # 1ª passada: contar operadores de cada arquivo para classificar.
    contagens = {}
    for path in json_files:
        try:
            contagens[path] = len(extrair_operadores(path))
        except Exception as e:
            print(f"  [aviso] falha ao ler {os.path.basename(path)}: {e}; ignorado")

    if not contagens:
        print("Nenhum arquivo pôde ser lido.")
        sys.exit(1)

    # O search space é a classe com MAIS operadores; o resto são fronteiras.
    max_ops = max(contagens.values())
    arquivos_search = [p for p, n in contagens.items() if n == max_ops]
    arquivos_pareto = [p for p, n in contagens.items() if n < max_ops]

    print(f"  search space  = {max_ops} operadores  ({len(arquivos_search)} arquivo(s))")
    if arquivos_pareto:
        n_fronteira = contagens[arquivos_pareto[0]]
        print(f"  fronteiras    = {n_fronteira} operadores  ({len(arquivos_pareto)} arquivo(s))")
    else:
        print("  [aviso] nenhum arquivo de fronteira de Pareto encontrado; só o search space será plotado")

    # Base do gráfico: PRIMEIRO arquivo de search space (feito uma única vez).
    base_path = sorted(arquivos_search)[0]
    print(f"\nSearch space base: {os.path.basename(base_path)}")
    search_ops = extrair_operadores(base_path)

    # Mantém só operadores com cost/quality válidos.
    pontos = {
        oid: (c, q)
        for oid, (c, q) in search_ops.items()
        if c is not None and q is not None
    }
    descartados = len(search_ops) - len(pontos)
    if descartados:
        print(f"  [aviso] {descartados} operador(es) sem cost/quality no search space foram ignorados")

    # União dos ids de todas as fronteiras de Pareto.
    ids_pareto = set()
    for path in arquivos_pareto:
        ids_pareto.update(extrair_operadores(path).keys())

    # Separa os pontos em "search space" (azul) e "fronteira" (vermelho).
    na_fronteira = [oid for oid in pontos if oid in ids_pareto]
    fora = [oid for oid in pontos if oid not in ids_pareto]

    sem_correspondencia = ids_pareto - set(pontos.keys())
    if sem_correspondencia:
        print(f"  [aviso] {len(sem_correspondencia)} id(s) de fronteira não encontrados no search space")

    print(f"\nPontos totais: {len(pontos)} | destacados (Pareto): {len(na_fronteira)}")

    # --- Plot ------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(9, 7))

    # Azul claro: search space inteiro (desenhado uma vez).
    ax.scatter(
        [pontos[o][0] for o in fora],
        [pontos[o][1] for o in fora],
        s=55, color="#a6cee3", edgecolors="#5a8fb0", linewidths=0.4,
        alpha=0.85, label="Search space", zorder=1,
    )

    # Vermelho por cima: operadores presentes em alguma fronteira de Pareto.
    if na_fronteira:
        ax.scatter(
            [pontos[o][0] for o in na_fronteira],
            [pontos[o][1] for o in na_fronteira],
            s=70, color="#e31a1c", edgecolors="black", linewidths=0.6,
            alpha=0.95, label="Fronteira de Pareto", zorder=3,
        )

    ax.set_xlabel("Cost")
    ax.set_ylabel("Quality")
    ax.set_title("Espaço de busca de operadores físicos — quality x cost")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(directory, "search_space_pareto.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSalvo: {out_path}\nPronto.")


if __name__ == "__main__":
    main()