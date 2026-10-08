import os
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd

# 1. Configuração do diretório onde estão seus JSONs da imagem
PASTA_SEARCH_SPACES = "./search-spaces"  # Substitua pelo caminho correto na Morpheus se necessário

def extrair_metricas_palimpzest(caminho_arquivo):
    """
    Carrega o JSON e extrai as informações de custo e qualidade dos grupos de operadores.
    """
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        try:
            dados = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Erro ao ler {caminho_arquivo}: {e}")
            return None

    registros = []
    groups = dados.get("groups", {})
    
    for group_id, group_info in groups.items():
        # Tenta pegar custo e qualidade das propriedades do grupo
        props = group_info.get("properties", {})
        
        # O custo/qualidade costuma estar em "properties" no Palimpzest/Abacus
        # Substitua pelas chaves exatas se forem diferentes (ex: "estimated_cost", "plan_cost")
        custo = props.get("cost", props.get("estimated_cost", None))
        qualidade = props.get("quality", props.get("accuracy", None))
        
        # Fallback para os operadores se não achou no grupo
        operators = group_info.get("operators", [])
        nomes_operadores = [op.get("operator_class", "Op") for op in operators]
        label_operadores = " + ".join(nomes_operadores) if nomes_operadores else f"Grupo {group_id[-4:]}"
        
        if custo is None or qualidade is None:
            # Se as métricas estiverem dentro de cada operador individualmente:
            for op in operators:
                c = op.get("cost", op.get("estimated_cost", 0.0))
                q = op.get("quality", op.get("score", 0.0))
                registros.append({
                    "grupo": group_id,
                    "operador": op.get("operator_class", "Op"),
                    "custo": float(c),
                    "qualidade": float(q)
                })
        else:
            registros.append({
                "grupo": group_id,
                "operador": label_operadores,
                "custo": float(custo),
                "qualidade": float(qualidade)
            })
            
    return registros

def plotar_e_salvar(dados_plot, caminho_saida, timestamp):
    """
    Gera o gráfico de dispersão Custo x Qualidade e salva como imagem.
    """
    if not dados_plot:
        return
        
    df = pd.DataFrame(dados_plot)
    
    plt.figure(figsize=(10, 6))
    
    # Plota os pontos dos planos
    scatter = plt.scatter(
        df['custo'], 
        df['qualidade'], 
        alpha=0.8, 
        s=120, 
        c='darkblue', 
        edgecolors='black',
        zorder=3
    )
    
    # Anota os nomes dos operadores em cada ponto
    for _, row in df.iterrows():
        plt.annotate(
            row['operador'],
            (row['custo'], row['qualidade']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3),
            zorder=4
        )
        
    plt.title(f"Trade-off Espaço de Busca - Teste: {timestamp}", fontsize=12, fontweight='bold')
    plt.xlabel("Custo Estimado", fontsize=10)
    plt.ylabel("Qualidade / Utilidade", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
    print(f"Gráfico gerado: {caminho_saida}")

def processar_todos_os_arquivos():
    """
    Varre a pasta de search spaces, identifica novos arquivos e gera os gráficos correspondentes.
    """
    arquivos_json = glob.glob(os.path.join(PASTA_SEARCH_SPACES, "search_space_*.json"))
    
    if not arquivos_json:
        print(f"Nenhum arquivo JSON encontrado no diretório: {PASTA_SEARCH_SPACES}")
        return
        
    print(f"Encontrados {len(arquivos_json)} arquivos de espaço de busca. Iniciando processamento...")
    
    for arq in arquivos_json:
        # Extrai o timestamp do nome do arquivo para usar no título e salvar
        # Ex: search_space_20260411_014006_73... -> 20260411_014006
        nome_base = os.path.basename(arq)
        partes = nome_base.split('_')
        timestamp = f"{partes[2]}_{partes[3]}" if len(partes) >= 4 else "desconhecido"
        
        caminho_grafico = arq.replace(".json", ".png")
        
        # Se o gráfico já existe para este teste, pula para economizar processamento
        if os.path.exists(caminho_grafico):
            continue
            
        dados = extrair_metricas_palimpzest(arq)
        if dados:
            plotar_e_salvar(dados, caminho_grafico, timestamp)

if __name__ == "__main__":
    processar_todos_os_arquivos()