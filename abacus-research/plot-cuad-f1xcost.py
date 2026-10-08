"""
plot_f1_vs_cost.py
------------------
Percorre um diretório com logs de execuções do Abacus/CUAD e plota
3 gráficos de qualidade (F1-score) em função do custo:
  - F1 x Optimization Cost
  - F1 x Plan Execution Cost
  - F1 x Total Execution Cost

Cada gráfico discrimina as estratégias: mab / bay / cxtbay.
Cada execução vira um ponto individual — sem linhas, sem agrupamento.

Uso:
    python plot_f1_vs_cost.py <caminho_para_o_diretorio_com_logs>
"""

import os
import re
import glob
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


def load_data(results_dir):
    patterns = {
        "strategy":   re.compile(r"Sentinel execution strategy:\s+(\S+)"),
        "F1":         re.compile(r"F1:\s+([\d.]+)"),
        "opt_cost":   re.compile(r"Optimization cost:\s+([\d.]+)"),
        "plan_cost":  re.compile(r"Plan execution cost:\s+([\d.]+)"),
        "total_cost": re.compile(r"Total execution cost:\s+([\d.]+)"),
    }

    records = []

    filepaths = glob.glob(os.path.join(results_dir, "*"))
    if not filepaths:
        print(f"[!] Nenhum arquivo encontrado em: {results_dir}")
        return records

    for filepath in sorted(filepaths):
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        m_strat = patterns["strategy"].search(content)
        m_f1    = patterns["F1"].search(content)
        m_opt   = patterns["opt_cost"].search(content)
        m_plan  = patterns["plan_cost"].search(content)
        m_total = patterns["total_cost"].search(content)

        if not all([m_strat, m_f1, m_opt, m_plan, m_total]):
            print(f"[~] Campos faltando em {os.path.basename(filepath)}, pulando.")
            continue

        records.append({
            "strategy":   m_strat.group(1).lower(),
            "f1":         float(m_f1.group(1)),
            "opt_cost":   float(m_opt.group(1)),
            "plan_cost":  float(m_plan.group(1)),
            "total_cost": float(m_total.group(1)),
        })

    print(f"[+] {len(records)} execuções carregadas.")
    return records


def plot_f1_vs_cost(results_dir):
    records = load_data(results_dir)
    if not records:
        return

    cost_axes = [
        ("opt_cost",   "Optimization Cost (USD)"),
        ("plan_cost",  "Plan Execution Cost (USD)"),
        ("total_cost", "Total Execution Cost (USD)"),
    ]

    strategies = ["mab", "bay", "cxtbay"]
    colors     = {"mab": "tab:blue", "bay": "tab:orange", "cxtbay": "tab:green"}
    markers    = {"mab": "o",        "bay": "s",          "cxtbay": "^"}
    labels     = {"mab": "MAB",      "bay": "BAY",        "cxtbay": "CXT-BAY"}

    # Agrupa pontos por estratégia
    by_strat = defaultdict(list)
    for r in records:
        by_strat[r["strategy"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "F1-Score vs. Custo por Estratégia",
        fontsize=14, fontweight="bold"
    )

    for ax, (cost_key, xlabel) in zip(axes, cost_axes):
        ax.set_title(xlabel, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("F1-Score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)

        all_f1 = []
        for strat in strategies:
            pts = by_strat.get(strat, [])
            if not pts:
                continue

            xs = [r[cost_key] for r in pts]
            ys = [r["f1"]     for r in pts]
            all_f1.extend(ys)

            ax.scatter(
                xs, ys,
                color=colors[strat],
                marker=markers[strat],
                s=55, alpha=0.85, zorder=5,
                label=labels[strat]
            )

        # Zoom automático no eixo Y com margem de 10%
        if all_f1:
            ymin, ymax = min(all_f1), max(all_f1)
            margin = (ymax - ymin) * 0.10 if ymax > ymin else 0.02
            ax.set_ylim(ymin - margin, ymax + margin)

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title="Strategy", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = os.path.join(results_dir, "f1_vs_cost.png")
    plt.savefig(output_path, dpi=300)
    print(f"[+] Gráficos salvos em: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_f1_vs_cost(sys.argv[1])
    else:
        print("Uso:  python plot-cuad-f1xcost.py <diretório_com_logs>")
        print("Ex.:  python plot-cuad-f1xcost.py ./cuad_logs")