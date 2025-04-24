import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

# Configuração para visualizações
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.family'] = 'DejaVu Sans'

# Assumindo que df_consolidado já está carregado
# As análises serão executadas neste DataFrame

# -------------------- 1. ANÁLISE DE TENDÊNCIAS TEMPORAIS --------------------

def analise_tendencias_temporais(df):
    print("Iniciando análise de tendências temporais...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/tendencias', exist_ok=True)
    
    # Agrupar por Produto e Ano para calcular o rendimento médio
    rendimento_medio = df.groupby(['Produto', 'Ano'])['Rendimento_KgPorHectare'].mean().reset_index()
    
    # Obter a lista de produtos únicos
    produtos = rendimento_medio['Produto'].unique()
    
    # Dicionário para armazenar pontos de inflexão
    pontos_inflexao = {}
    
    # Criar gráfico de linha para cada produto
    for produto in produtos:
        dados_produto = rendimento_medio[rendimento_medio['Produto'] == produto]
        
        plt.figure(figsize=(14, 8))
        plt.plot(dados_produto['Ano'], dados_produto['Rendimento_KgPorHectare'], 
                 marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Aplicar suavização usando LOWESS para identificar tendência
        filtered = lowess(dados_produto['Rendimento_KgPorHectare'].values, 
                          dados_produto['Ano'].values, frac=0.3)
        
        # Plotar linha de tendência
        plt.plot(dados_produto['Ano'], filtered[:, 1], 'r--', linewidth=2, 
                 label='Tendência (LOWESS)')
        
        # Identificar pontos de inflexão (anos com mudanças significativas)
        # Calcular a taxa de variação anual
        dados_produto['Variacao'] = dados_produto['Rendimento_KgPorHectare'].pct_change() * 100
        
        # Identificar anos com variação acima de um limiar (por exemplo, 15%)
        limiar_variacao = 15
        anos_inflexao = dados_produto[abs(dados_produto['Variacao']) > limiar_variacao]
        
        # Marcar os pontos de inflexão no gráfico
        if not anos_inflexao.empty:
            plt.plot(anos_inflexao['Ano'], anos_inflexao['Rendimento_KgPorHectare'], 
                     'ro', markersize=12, label='Pontos de Inflexão')
            
            # Adicionar anotações para cada ponto de inflexão
            for _, row in anos_inflexao.iterrows():
                plt.annotate(f"{int(row['Ano'])}: {row['Variacao']:.1f}%", 
                             xy=(row['Ano'], row['Rendimento_KgPorHectare']),
                             xytext=(10, 0), textcoords='offset points',
                             fontsize=11, fontweight='bold')
            
            # Armazenar anos de inflexão no dicionário
            pontos_inflexao[produto] = anos_inflexao[['Ano', 'Variacao']].values.tolist()
        
        plt.title(f'Evolução do Rendimento Médio de {produto} (1990-2022)', fontsize=16)
        plt.xlabel('Ano', fontsize=14)
        plt.ylabel('Rendimento (Kg/Hectare)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'figuras/tendencias/tendencia_{produto.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
    
    # Criar tabela resumo com os pontos de inflexão
    print("\nPontos de Inflexão Identificados (Anos com Variação > 15%):")
    for produto, anos in pontos_inflexao.items():
        if anos:
            print(f"\n{produto}:")
            for ano, variacao in anos:
                print(f"  - {int(ano)}: {variacao:.1f}%")
        else:
            print(f"\n{produto}: Nenhum ponto de inflexão significativo identificado.")
    
    return pontos_inflexao

# -------------------- 2. COMPARATIVOS REGIONAIS --------------------

def comparativos_regionais(df):
    print("\nIniciando análise de comparativos regionais...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/regionais', exist_ok=True)
    
    # Agrupar por Mesorregião e Produto para calcular o rendimento médio
    rendimento_por_regiao = df.groupby(['Mesorregião', 'Produto'])['Rendimento_KgPorHectare'].mean().reset_index()
    
    # Obter a lista de produtos únicos
    produtos = rendimento_por_regiao['Produto'].unique()
    
    # Dicionário para armazenar rankings
    rankings = {}
    
    # Para cada produto, criar um ranking das mesorregiões mais produtivas
    for produto in produtos:
        dados_produto = rendimento_por_regiao[rendimento_por_regiao['Produto'] == produto]
        ranking = dados_produto.sort_values('Rendimento_KgPorHectare', ascending=False).reset_index(drop=True)
        ranking.index = ranking.index + 1  # Ajustar o índice para começar em 1
        
        # Armazenar no dicionário
        rankings[produto] = ranking
        
        # Imprimir o ranking
        print(f"\nRanking das Mesorregiões mais Produtivas para {produto}:")
        print(ranking[['Mesorregião', 'Rendimento_KgPorHectare']].head(10))
        
        # Criar gráfico de barras horizontais para o top 10
        plt.figure(figsize=(14, 10))
        sns.barplot(x='Rendimento_KgPorHectare', y='Mesorregião', 
                    data=ranking.head(10), palette='viridis')
        plt.title(f'Top 10 Mesorregiões mais Produtivas - {produto}', fontsize=16)
        plt.xlabel('Rendimento Médio (Kg/Hectare)', fontsize=14)
        plt.ylabel('Mesorregião', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figuras/regionais/ranking_{produto.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
    
    # Criar mapas de calor mostrando a distribuição espacial da produtividade
    # Precisamos de uma tabela pivotada com mesorregiões nas linhas e produtos nas colunas
    tabela_calor = rendimento_por_regiao.pivot(index='Mesorregião', columns='Produto', values='Rendimento_KgPorHectare')
    
    # Normalizar os valores para cada produto para melhor visualização
    tabela_normalizada = tabela_calor.copy()
    for col in tabela_normalizada.columns:
        tabela_normalizada[col] = (tabela_normalizada[col] - tabela_normalizada[col].min()) / \
                                 (tabela_normalizada[col].max() - tabela_normalizada[col].min())
    
    # Criar o mapa de calor
    plt.figure(figsize=(16, 14))
    cmap = LinearSegmentedColormap.from_list('GreenBlue', 
                                            ['#ffffcc', '#41b6c4', '#253494'], 
                                            N=256)
    sns.heatmap(tabela_normalizada, cmap=cmap, linewidths=0.5, 
                linecolor='gray', annot=False, vmin=0, vmax=1)
    plt.title('Distribuição Espacial da Produtividade por Mesorregião e Cultura (Normalizada)', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('figuras/regionais/mapa_calor_produtividade.png', dpi=300)
    plt.close()
    
    return rankings

# -------------------- 3. CORRELAÇÕES ENTRE VARIÁVEIS --------------------

def analise_correlacoes(df):
    print("\nIniciando análise de correlações entre variáveis...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/correlacoes', exist_ok=True)
    
    # Obter a lista de produtos únicos
    produtos = df['Produto'].unique()
    
    # Dicionário para armazenar correlações
    correlacoes = {}
    
    # Para cada produto, analisar correlações
    for produto in produtos:
        dados_produto = df[df['Produto'] == produto]
        
        # Calcular correlação entre área plantada e rendimento
        corr_area_rend = dados_produto['Area_Plantada_Hectares'].corr(dados_produto['Rendimento_KgPorHectare'])
        
        # Calcular correlação entre valor da produção e rendimento
        corr_valor_rend = dados_produto['Valor_Produzido_Mil_Reais'].corr(dados_produto['Rendimento_KgPorHectare'])
        
        # Armazenar no dicionário
        correlacoes[produto] = {
            'Area_Rendimento': corr_area_rend,
            'Valor_Rendimento': corr_valor_rend
        }
        
        # Criar gráficos de dispersão
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Gráfico 1: Área plantada vs. Rendimento
        sns.scatterplot(x='Area_Plantada_Hectares', y='Rendimento_KgPorHectare', 
                        data=dados_produto, ax=ax1, alpha=0.6, s=50)
        
        # Adicionar linha de tendência
        x = dados_produto['Area_Plantada_Hectares']
        y = dados_produto['Rendimento_KgPorHectare']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), "r--", linewidth=2)
        
        ax1.set_title(f'Relação entre Área Plantada e Rendimento\n{produto} (r = {corr_area_rend:.2f})', fontsize=14)
        ax1.set_xlabel('Área Plantada (Hectares)', fontsize=12)
        ax1.set_ylabel('Rendimento (Kg/Hectare)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Valor da produção vs. Rendimento
        sns.scatterplot(x='Valor_Produzido_Mil_Reais', y='Rendimento_KgPorHectare', 
                        data=dados_produto, ax=ax2, alpha=0.6, s=50)
        
        # Adicionar linha de tendência
        x = dados_produto['Valor_Produzido_Mil_Reais']
        y = dados_produto['Rendimento_KgPorHectare']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax2.plot(x, p(x), "r--", linewidth=2)
        
        ax2.set_title(f'Relação entre Valor da Produção e Rendimento\n{produto} (r = {corr_valor_rend:.2f})', fontsize=14)
        ax2.set_xlabel('Valor da Produção (Mil Reais)', fontsize=12)
        ax2.set_ylabel('Rendimento (Kg/Hectare)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figuras/correlacoes/correlacao_{produto.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()

    # Adicionar análise de correlações com variáveis climáticas
    var_climaticas = ['precipitacao_total_anual', 'radiacao_global_media', 
                     'temperatura_bulbo_media', 'vento_velocidade_media']
    
    for produto in produtos:
        dados_produto = df[df['Produto'] == produto]
        
        # Criar matriz de correlação
        cols_analise = var_climaticas + ['Rendimento_KgPorHectare']
        matriz_corr = dados_produto[cols_analise].corr()
        
        # Visualizar matriz de correlação
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(matriz_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=.5)
        
        plt.title(f'Correlações entre Rendimento e Variáveis Climáticas - {produto}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'figuras/correlacoes/clima_{produto.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
        
        # Adicionar ao dicionário de correlações
        for var in var_climaticas:
            correlacoes[produto][f'{var}_Rendimento'] = \
                matriz_corr.loc['Rendimento_KgPorHectare', var]
    
    # Imprimir tabela resumo de correlações
    print("\nResumo das Correlações:")
    for produto, corrs in correlacoes.items():
        print(f"\n{produto}:")
        print(f"  - Área Plantada vs Rendimento: {corrs['Area_Rendimento']:.3f}")
        print(f"  - Valor da Produção vs Rendimento: {corrs['Valor_Rendimento']:.3f}")
        for var in var_climaticas:
            var_key = f'{var}_Rendimento'
            if var_key in corrs:
                print(f"  - {var} vs Rendimento: {corrs[var_key]:.3f}")
    
    return correlacoes

# -------------------- 4. ANÁLISE DE VOLATILIDADE --------------------

def analise_volatilidade(df):
    print("\nIniciando análise de volatilidade...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/volatilidade', exist_ok=True)
    
    # Agrupar por Produto, Mesorregião e Ano
    dados_agrupados = df.groupby(['Produto', 'Mesorregião', 'Ano'])['Rendimento_KgPorHectare'].mean().reset_index()
    
    # Calcular coeficiente de variação por Produto e Mesorregião
    cv_por_produto_regiao = dados_agrupados.groupby(['Produto', 'Mesorregião']).agg(
        Rendimento_Medio=('Rendimento_KgPorHectare', 'mean'),
        Desvio_Padrao=('Rendimento_KgPorHectare', 'std')
    ).reset_index()
    
    # Calcular o coeficiente de variação (CV = desvio padrão / média)
    cv_por_produto_regiao['CV'] = (cv_por_produto_regiao['Desvio_Padrao'] / 
                                  cv_por_produto_regiao['Rendimento_Medio']) * 100
    
    # Ordenar por CV
    cv_por_produto_regiao = cv_por_produto_regiao.sort_values('CV')
    
    # Identificar as culturas mais estáveis e instáveis
    culturas_volatilidade = cv_por_produto_regiao.groupby('Produto').agg(
        CV_Medio=('CV', 'mean')
    ).sort_values('CV_Medio').reset_index()
    
    print("\nVolatilidade por Cultura (Coeficiente de Variação Médio):")
    print(culturas_volatilidade)
    
    # Criar gráfico de barras para volatilidade das culturas
    plt.figure(figsize=(12, 8))
    sns.barplot(x='CV_Medio', y='Produto', data=culturas_volatilidade, palette='coolwarm')
    plt.title('Volatilidade do Rendimento por Cultura', fontsize=16)
    plt.xlabel('Coeficiente de Variação Médio (%)', fontsize=14)
    plt.ylabel('Cultura', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figuras/volatilidade/volatilidade_culturas.png', dpi=300)
    plt.close()
    
    # Identificar as regiões mais estáveis e instáveis para cada produto
    for produto in df['Produto'].unique():
        cv_produto = cv_por_produto_regiao[cv_por_produto_regiao['Produto'] == produto]
        
        # Regiões mais estáveis (menor CV)
        regioes_estaveis = cv_produto.sort_values('CV').head(10)
        
        # Regiões mais instáveis (maior CV)
        regioes_instaveis = cv_produto.sort_values('CV', ascending=False).head(10)
        
        print(f"\nRegiões mais Estáveis para {produto}:")
        print(regioes_estaveis[['Mesorregião', 'CV']])
        
        print(f"\nRegiões mais Instáveis para {produto}:")
        print(regioes_instaveis[['Mesorregião', 'CV']])
        
        # Criar gráfico comparativo
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Gráfico das regiões mais estáveis
        sns.barplot(x='CV', y='Mesorregião', data=regioes_estaveis, 
                   palette='Blues_r', ax=ax1)
        ax1.set_title(f'Regiões mais Estáveis para {produto}', fontsize=14)
        ax1.set_xlabel('Coeficiente de Variação (%)', fontsize=12)
        ax1.set_ylabel('Mesorregião', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico das regiões mais instáveis
        sns.barplot(x='CV', y='Mesorregião', data=regioes_instaveis, 
                   palette='Reds', ax=ax2)
        ax2.set_title(f'Regiões mais Instáveis para {produto}', fontsize=14)
        ax2.set_xlabel('Coeficiente de Variação (%)', fontsize=12)
        ax2.set_ylabel('Mesorregião', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figuras/volatilidade/estabilidade_{produto.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
    
    return cv_por_produto_regiao, culturas_volatilidade

# -------------------- 5. TAXONOMIA DE MESORREGIÕES --------------------

def taxonomia_mesorregioes(df):
    print("\nIniciando taxonomia de mesorregiões...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/taxonomia', exist_ok=True)
    
    # Preparar os dados para clustering
    # Pivotear para obter mesorregiões nas linhas e produtos nas colunas
    rendimento_pivot = df.pivot_table(
        index='Mesorregião', 
        columns='Produto', 
        values='Rendimento_KgPorHectare',
        aggfunc='mean'
    ).fillna(0)  # Preencher NaN com 0
    
    # Normalizar os dados para o clustering
    scaler = StandardScaler()
    rendimento_scaled = scaler.fit_transform(rendimento_pivot)
    
    # Determinar o número ótimo de clusters usando o método do cotovelo
    inertias = []
    silhouettes = []
    range_n_clusters = range(2, 11)
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(rendimento_scaled)
        inertias.append(kmeans.inertia_)
        
        # Calcular o score de silhueta
        if n_clusters > 1:  # Silhueta não é definida para 1 cluster
            silhouette_avg = silhouette_score(rendimento_scaled, cluster_labels)
            silhouettes.append(silhouette_avg)
        else:
            silhouettes.append(0)
    
    # Plotar o gráfico do cotovelo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    ax1.plot(range_n_clusters, inertias, 'o-', linewidth=2, markersize=8)
    ax1.set_title('Método do Cotovelo', fontsize=14)
    ax1.set_xlabel('Número de Clusters', fontsize=12)
    ax1.set_ylabel('Inércia', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range_n_clusters, silhouettes, 'o-', linewidth=2, markersize=8)
    ax2.set_title('Método da Silhueta', fontsize=14)
    ax2.set_xlabel('Número de Clusters', fontsize=12)
    ax2.set_ylabel('Score de Silhueta', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figuras/taxonomia/otimizacao_clusters.png', dpi=300)
    plt.close()
    
    # Escolher o número ótimo de clusters (você pode ajustar com base no gráfico)
    n_clusters = 4  # Valor padrão, ajuste com base nos resultados
    
    # Aplicar K-means com o número ótimo de clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rendimento_scaled)
    
    # Adicionar as labels de cluster ao dataframe
    rendimento_pivot['Cluster'] = cluster_labels
    
    # Analisar os clusters
    cluster_info = rendimento_pivot.groupby('Cluster').mean()
    
    # Identificar as características de cada cluster
    print("\nCaracterísticas dos Clusters:")
    for i in range(n_clusters):
        # Encontrar as culturas mais importantes para cada cluster
        culturas_cluster = cluster_info.loc[i].sort_values(ascending=False).index[:3].tolist()
        n_mesorregioes = (rendimento_pivot['Cluster'] == i).sum()
        
        print(f"\nCluster {i} - {n_mesorregioes} mesorregiões")
        print(f"  Principais culturas: {', '.join(culturas_cluster)}")
        print(f"  Mesorregiões no cluster:")
        for mesorregiao in rendimento_pivot[rendimento_pivot['Cluster'] == i].index[:5]:
            print(f"    - {mesorregiao}")
        if n_mesorregioes > 5:
            print(f"    - ... e mais {n_mesorregioes - 5} mesorregiões")
    
    # Visualizar os clusters em um mapa de calor
    # Ordenar o dataframe pelo cluster
    rendimento_pivot_sorted = rendimento_pivot.sort_values('Cluster')
    
    # Remover a coluna de cluster para o mapa de calor
    heatmap_data = rendimento_pivot_sorted.drop(columns=['Cluster'])
    
    # Normalizar os dados para melhor visualização
    for col in heatmap_data.columns:
        heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / \
                         (heatmap_data[col].max() - heatmap_data[col].min())
    
    # Criar o mapa de calor
    plt.figure(figsize=(16, 14))
    
    # Adicionar coluna de cluster para colorir as linhas
    row_colors = [f'C{i}' for i in rendimento_pivot_sorted['Cluster']]
    
    cmap = LinearSegmentedColormap.from_list('GreenBlue', 
                                            ['#ffffcc', '#41b6c4', '#253494'], 
                                            N=256)
    
    g = sns.clustermap(heatmap_data, figsize=(16, 14), cmap=cmap,
                     row_colors=row_colors, standard_scale=1, 
                     linewidths=0.1, linecolor='gray', 
                     yticklabels=1, xticklabels=1)
    
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=10, rotation=45, ha='right')
    
    plt.suptitle('Agrupamento de Mesorregiões por Padrões de Produtividade', fontsize=18, y=0.92)
    plt.savefig('figuras/taxonomia/clusters_mesorregioes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salvar os resultados do clustering
    resultados_cluster = rendimento_pivot[['Cluster']].copy()
    resultados_cluster.to_csv('taxonomia_mesorregioes.csv')
    
    return resultados_cluster, cluster_info

# -------------------- 6. SÉRIES TEMPORAIS AVANÇADAS --------------------

def series_temporais_avancadas(df):
    print("\nIniciando análise de séries temporais avançadas...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/series', exist_ok=True)
    
    # Agrupar por Produto e Ano para análise de séries temporais
    series_por_produto = df.groupby(['Produto', 'Ano'])['Rendimento_KgPorHectare'].mean().reset_index()
    
    # Para cada produto, realizar análise de séries temporais
    for produto in df['Produto'].unique():
        dados_produto = series_por_produto[series_por_produto['Produto'] == produto]
        
        # Ordenar por ano
        dados_produto = dados_produto.sort_values('Ano')
        
        # Converter para série temporal
        ts = pd.Series(dados_produto['Rendimento_KgPorHectare'].values, 
                      index=pd.to_datetime(dados_produto['Ano'], format='%Y'))
        
        # Verificar se há pontos suficientes para decomposição
        if len(ts) >= 2 * 4:  # Pelo menos 2 períodos completos (assumindo estacional de 4 anos)
            try:
                # Decomposição da série temporal
                decomposicao = seasonal_decompose(ts, model='additive', period=4)  # Período de 4 anos
                
                # Plotar a decomposição
                fig, axes = plt.subplots(4, 1, figsize=(14, 16))
                decomposicao.observed.plot(ax=axes[0], title='Série Original')
                decomposicao.trend.plot(ax=axes[1], title='Tendência')
                decomposicao.seasonal.plot(ax=axes[2], title='Sazonalidade')
                decomposicao.resid.plot(ax=axes[3], title='Resíduos')
                
                for ax in axes:
                    ax.set_xlabel('Ano')
                    ax.set_ylabel('Rendimento (Kg/Hectare)')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'figuras/series/decomposicao_{produto.replace(" ", "_").lower()}.png', dpi=300)
                plt.close()
                
                # Detecção de outliers nos resíduos
                residuos = decomposicao.resid.dropna()
                
                # Calcular limites para outliers (método IQR)
                Q1 = residuos.quantile(0.25)
                Q3 = residuos.quantile(0.75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                # Identificar outliers
                outliers = residuos[(residuos < limite_inferior) | (residuos > limite_superior)]
                
                if not outliers.empty:
                    print(f"\nOutliers detectados para {produto}:")
                    for data, valor in outliers.items():
                        print(f"  - {data.year}: {valor:.2f}")
                    
                    # Marcar outliers na série original
                    plt.figure(figsize=(14, 8))
                    plt.plot(ts.index, ts.values, 'b-', linewidth=2, label='Série Original')
                    plt.scatter(outliers.index, ts[outliers.index], color='red', s=100, 
                                label='Outliers', zorder=5)
                    
                    for idx in outliers.index:
                        plt.annotate(f"{idx.year}", 
                                    xy=(idx, ts[idx]),
                                    xytext=(10, 0), textcoords='offset points',
                                    fontsize=12, fontweight='bold')
                    
                    plt.title(f'Detecção de Outliers na Série Temporal - {produto}', fontsize=16)
                    plt.xlabel('Ano', fontsize=14)
                    plt.ylabel('Rendimento (Kg/Hectare)', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=12)
                    plt.tight_layout()
                    plt.savefig(f'figuras/series/outliers_{produto.replace(" ", "_").lower()}.png', dpi=300)
                    plt.close()
            except Exception as e:
                print(f"Não foi possível realizar a decomposição para {produto}: {str(e)}")
    
    return series_por_produto

# -------------------- 7. INDICADORES DE ESPECIALIZAÇÃO REGIONAL --------------------

def indicadores_especializacao(df):
    print("\nIniciando análise de indicadores de especialização regional...")
    
    # Criar diretório para salvar as figuras
    import os
    os.makedirs('figuras/especializacao', exist_ok=True)
    
    # Calcular a participação de cada cultura por mesorregião com base na área plantada
    area_total_por_regiao = df.groupby(['Mesorregião', 'Ano'])['Area_Plantada_Hectares'].sum().reset_index()
    area_total_por_regiao.rename(columns={'Area_Plantada_Hectares': 'Area_Total'}, inplace=True)
    
    # Mesclar com os dados originais
    dados_merged = pd.merge(df, area_total_por_regiao, on=['Mesorregião', 'Ano'])
    
    # Calcular participação de cada cultura
    dados_merged['Participacao'] = dados_merged['Area_Plantada_Hectares'] / dados_merged['Area_Total']
    
    # Calcular o índice de especialização (IER - Índice de Especialização Regional)
    # Primeiro, calcular a participação média nacional de cada cultura
    area_total_nacional = df.groupby('Ano')['Area_Plantada_Hectares'].sum().reset_index()
    area_total_nacional.rename(columns={'Area_Plantada_Hectares': 'Area_Total_Nacional'}, inplace=True)
    
    area_por_cultura_nacional = df.groupby(['Produto', 'Ano'])['Area_Plantada_Hectares'].sum().reset_index()
    area_por_cultura_nacional.rename(columns={'Area_Plantada_Hectares': 'Area_Cultura_Nacional'}, inplace=True)
    
    # Mesclar os totais nacionais
    dados_nacional = pd.merge(area_por_cultura_nacional, area_total_nacional, on='Ano')
    dados_nacional['Participacao_Nacional'] = dados_nacional['Area_Cultura_Nacional'] / dados_nacional['Area_Total_Nacional']
    
    # Mesclar com os dados regionais
    dados_completos = pd.merge(dados_merged, 
                             dados_nacional[['Produto', 'Ano', 'Participacao_Nacional']], 
                             on=['Produto', 'Ano'])
    
    # Calcular o IER (quociente locacional)
    dados_completos['IER'] = dados_completos['Participacao'] / dados_completos['Participacao_Nacional']
    
    # Agrupar por Mesorregião e Produto para ter uma visão geral
    ier_medio = dados_completos.groupby(['Mesorregião', 'Produto'])['IER'].mean().reset_index()
    
    # Analisar os IERs por mesorregião
    # Considerar IER > 1 como indicativo de especialização
    especializacoes = ier_medio[ier_medio['IER'] > 1].sort_values(['Mesorregião', 'IER'], ascending=[True, False])
    
    # Contar o número de culturas em que cada mesorregião é especializada
    contagem_especializacoes = especializacoes.groupby('Mesorregião').size().reset_index(name='Num_Especializacoes')
    
    # Ordenar por número de especializações (menor = mais especializada)
    contagem_especializacoes = contagem_especializacoes.sort_values('Num_Especializacoes')
    
    print("\nNúmero de Culturas em que cada Mesorregião é Especializada:")
    print(contagem_especializacoes.head(10))
    
    # Mostrar as mesorregiões mais especializadas em cada cultura
    for produto in df['Produto'].unique():
        ier_produto = ier_medio[ier_medio['Produto'] == produto].sort_values('IER', ascending=False)
        
        print(f"\nMesorregiões mais Especializadas em {produto}:")
        print(ier_produto.head(5)[['Mesorregião', 'IER']])
        
        # Criar gráfico das top 10 mesorregiões mais especializadas
        plt.figure(figsize=(14, 8))
        top_10 = ier_produto.head(10)
        sns.barplot(x='IER', y='Mesorregião', data=top_10, palette='viridis')
        plt.title(f'Top 10 Mesorregiões Especializadas em {produto}', fontsize=16)
        plt.xlabel('Índice de Especialização Regional (IER)', fontsize=14)
        plt.ylabel('Mesorregião', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figuras/especializacao/especializacao_{produto.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
    
    # Analisar a evolução da diversificação agrícola nas mesorregiões
    # Calcular o índice de diversificação (HHI - Índice de Herfindahl-Hirschman Invertido)
    # HHI = 1 - Σ(participação_i²), onde i é cada cultura
    diversificacao = dados_merged.groupby(['Mesorregião', 'Ano']).apply(
        lambda x: 1 - sum(x['Participacao'] ** 2)
    ).reset_index(name='Indice_Diversificacao')
    
    # Calcular a média do índice de diversificação para cada mesorregião
    diversificacao_media = diversificacao.groupby('Mesorregião')['Indice_Diversificacao'].mean().reset_index()
    diversificacao_media = diversificacao_media.sort_values('Indice_Diversificacao', ascending=False)
    
    print("\nMesorregiões com Maior Diversificação Agrícola:")
    print(diversificacao_media.head(10))
    
    print("\nMesorregiões com Menor Diversificação Agrícola:")
    print(diversificacao_media.tail(10))
    
    # Criar gráfico com as mesorregiões mais e menos diversificadas
    plt.figure(figsize=(14, 10))
    
    # Top 10 mais diversificadas
    plt.subplot(2, 1, 1)
    sns.barplot(x='Indice_Diversificacao', y='Mesorregião', 
               data=diversificacao_media.head(10), palette='Blues_r')
    plt.title('Top 10 Mesorregiões com Maior Diversificação Agrícola', fontsize=14)
    plt.xlabel('Índice de Diversificação', fontsize=12)
    plt.ylabel('Mesorregião', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Top 10 menos diversificadas
    plt.subplot(2, 1, 2)
    sns.barplot(x='Indice_Diversificacao', y='Mesorregião', 
               data=diversificacao_media.tail(10).iloc[::-1], palette='Reds_r')
    plt.title('Top 10 Mesorregiões com Menor Diversificação Agrícola', fontsize=14)
    plt.xlabel('Índice de Diversificação', fontsize=12)
    plt.ylabel('Mesorregião', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figuras/especializacao/diversificacao_mesorregioes.png', dpi=300)
    plt.close()
    
    # Analisar a evolução temporal da diversificação
    # Calcular o índice de diversificação médio nacional por ano
    diversificacao_nacional = diversificacao.groupby('Ano')['Indice_Diversificacao'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    plt.plot(diversificacao_nacional['Ano'], diversificacao_nacional['Indice_Diversificacao'], 
            'o-', linewidth=2, markersize=8)
    plt.title('Evolução da Diversificação Agrícola Nacional (1990-2022)', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Índice de Diversificação Médio', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figuras/especializacao/evolucao_diversificacao_nacional.png', dpi=300)
    plt.close()
    
    # Selecionar algumas mesorregiões para análise da evolução da diversificação
    # Selecionamos as 5 mais diversificadas e as 5 menos diversificadas
    top_diversificadas = diversificacao_media.head(5)['Mesorregião'].tolist()
    menos_diversificadas = diversificacao_media.tail(5)['Mesorregião'].tolist()
    mesorregioes_selecionadas = top_diversificadas + menos_diversificadas
    
    # Filtrar dados para estas mesorregiões
    evolucao_diversificacao = diversificacao[diversificacao['Mesorregião'].isin(mesorregioes_selecionadas)]
    
    plt.figure(figsize=(14, 10))
    
    for mesorregiao in mesorregioes_selecionadas:
        dados_mesorregiao = evolucao_diversificacao[evolucao_diversificacao['Mesorregião'] == mesorregiao]
        plt.plot(dados_mesorregiao['Ano'], dados_mesorregiao['Indice_Diversificacao'], 
                'o-', linewidth=2, markersize=6, label=mesorregiao)
    
    plt.title('Evolução da Diversificação Agrícola por Mesorregião (1990-2022)', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Índice de Diversificação', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figuras/especializacao/evolucao_diversificacao_regional.png', dpi=300)
    plt.close()
    
    return ier_medio, diversificacao_media

# -------------------- 8. FUNÇÃO PRINCIPAL E EXECUÇÃO --------------------

def executar_analises(df):
    """
    Função principal para executar todas as análises
    """
    # Criar diretório para figuras
    import os
    os.makedirs('figuras', exist_ok=True)
    
    # Executar análises
    print("Iniciando execução das análises...")
    
    # 1. Análise de tendências temporais
    pontos_inflexao = analise_tendencias_temporais(df)
    
    # 2. Comparativos regionais
    rankings = comparativos_regionais(df)
    
    # 3. Correlações entre variáveis
    correlacoes = analise_correlacoes(df)
    
    # 4. Análise de volatilidade
    cv_por_produto_regiao, culturas_volatilidade = analise_volatilidade(df)
    
    # 5. Taxonomia de mesorregiões
    resultados_cluster, cluster_info = taxonomia_mesorregioes(df)
    
    # 6. Séries temporais avançadas
    series_por_produto = series_temporais_avancadas(df)
    
    # 7. Indicadores de especialização regional
    ier_medio, diversificacao_media = indicadores_especializacao(df)
    
    print("\nTodas as análises foram concluídas com sucesso!")
    print("Os resultados foram salvos na pasta 'figuras' e respectivas subpastas.")
    
    return {
        'pontos_inflexao': pontos_inflexao,
        'rankings': rankings,
        'correlacoes': correlacoes,
        'volatilidade': {
            'cv_por_produto_regiao': cv_por_produto_regiao,
            'culturas_volatilidade': culturas_volatilidade
        },
        'taxonomia': {
            'resultados_cluster': resultados_cluster,
            'cluster_info': cluster_info
        },
        'series_temporais': series_por_produto,
        'especializacao': {
            'ier_medio': ier_medio,
            'diversificacao_media': diversificacao_media
        }
    }

# Exemplo de uso:
df_consolidado=pd.read_parquet('dados_meteo.parquet')
resultados = executar_analises(df_consolidado)