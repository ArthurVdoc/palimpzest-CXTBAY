import os
import re
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_metrics(results_dir):
    # Expressões regulares para extrair estratégia e budget
    strat_pattern = re.compile(r"Sentinel execution strategy:\s+(\w+)")
    budget_pattern = re.compile(r"Sample budget:\s+(\d+)")

    # Nomes das 9 métricas exatamente como aparecem no log
    metric_names = [
        "Precision", "Recall", "F1",
        "Optimization time", "Optimization cost",
        "Plan execution time", "Plan execution cost",
        "Total execution time", "Total execution cost"
    ]

    # Dicionário com os padrões Regex para cada métrica
    metric_patterns = {m: re.compile(rf"{m}:\s+([\d.]+)") for m in metric_names}

    # Estrutura de dados: data[metrica][estrategia][budget] = [valor1, valor2, ...]
    data = {m: defaultdict(lambda: defaultdict(list)) for m in metric_names}

    # Percorre todos os arquivos no diretório especificado
    filepaths = glob.glob(os.path.join(results_dir, "*"))
    
    if not filepaths:
        print(f"Nenhum arquivo encontrado no diretório: {results_dir}")
        return

    for filepath in filepaths:
        if not os.path.isfile(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        strat_match = strat_pattern.search(content)
        budget_match = budget_pattern.search(content)

        if strat_match and budget_match:
            strategy = strat_match.group(1)
            budget = int(budget_match.group(1))

            for m in metric_names:
                m_match = metric_patterns[m].search(content)
                if m_match:
                    data[m][strategy][budget].append(float(m_match.group(1)))

    # Configuração dos Gráficos
    strategies_to_plot = ["mab", "bay", "cxtbay"]
    colors = {"mab": "tab:blue", "bay": "tab:orange", "cxtbay": "tab:green"}

    # Cria uma grade 3x3 de gráficos
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Desempenho das Estratégias por Budget (Mediana ± IQR)', fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        ax.set_title(f"{metric} vs Budget (Median ± IQR)", fontsize=14)
        ax.set_xlabel("Sample Budget", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4) # Deixei o grid um pouco mais suave igual à imagem

        for strategy in strategies_to_plot:
            if strategy not in data[metric]:
                continue

            # Ordena os budgets para plotar a linha na sequência correta
            budgets = sorted(data[metric][strategy].keys())
            
            if not budgets:
                continue

            # Calcula a mediana, o 25º e o 75º percentis (IQR) para cada budget
            medians = [np.median(data[metric][strategy][b]) for b in budgets]
            p25 = [np.percentile(data[metric][strategy][b], 25) for b in budgets]
            p75 = [np.percentile(data[metric][strategy][b], 75) for b in budgets]

            # Define a cor baseada no dicionário
            color = colors.get(strategy, 'black')

            # Plota a linha principal (Mediana)
            ax.plot(budgets, medians, marker='o', linewidth=2, label=strategy.upper(), color=color)
            
            # Preenche a área do IQR (Sombra)
            ax.fill_between(budgets, p25, p75, alpha=0.2, color=color)

        # Adiciona a legenda apenas se houver linhas plotadas
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    # Ajusta o layout para que os títulos não se sobreponham
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salva o arquivo de imagem no mesmo diretório dos logs
    output_image_path = os.path.join(results_dir, "resultados_medianas_iqr.png")
    plt.savefig(output_image_path, dpi=300)
    print(f"\n[+] Análise concluída com sucesso!")
    print(f"[+] Gráfico salvo em: {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        plot_metrics(target_dir)
    else:
        print("Uso: python plot_metrics.py <caminho_para_o_diretorio_com_logs>")
        print("Exemplo: python plot_metrics.py ./saidas_testes")