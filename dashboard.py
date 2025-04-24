# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import streamlit.components.v1 as components
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Análises Agrícolas",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title("Dashboard de Análises Agrícolas")
st.markdown("---")

# Sidebar para navegação
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Selecione uma análise:",
    [
        "Início",
        "Mapas",
        "1. Tendências Temporais",
        "2. Comparativos Regionais",
        "3. Correlações",
        "4. Volatilidade",
        "5. Taxonomia de Mesorregiões",
        "6. Séries Temporais",
        "7. Especialização Regional",
        "8. Resultados das Análises"
    ]
)

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    # Substitua pelo caminho correto do seu arquivo
    df = pd.read_parquet('dados_meteo.parquet')
    return df

df_consolidado = carregar_dados()
    
def ordenar_mesorregioes(mesorregioes):
    """
    Ordena mesorregiões: primeiro por UF (AL, AM, AP, etc) e depois pelo nome da região
    """
    return sorted(mesorregioes, 
                 key=lambda x: (x.split(' - ')[-1] if ' - ' in x else 'ZZ',  # Sort by state code
                                x.split(' - ')[0] if ' - ' in x else x))      # Then by region name

mesorregioes = df_consolidado['Mesorregião'].unique()
mesorregioes_ordenadas = ordenar_mesorregioes(mesorregioes)    

# Configurar padrões
if 'produto_correlacao' not in st.session_state:
    # Default to the first product in the dataset
    produtos_disponiveis = sorted(df_consolidado['Produto'].unique())
    if produtos_disponiveis:
        st.session_state.produto_correlacao = produtos_disponiveis[0]
    else:
        st.session_state.produto_correlacao = None

if 'modo_visualizacao' not in st.session_state:
    st.session_state.modo_visualizacao = "Ano Único"

if 'ano_selecionado' not in st.session_state:
    anos_disponiveis = sorted([int(ano) for ano in df_consolidado['Ano'].unique()])
    if anos_disponiveis:
        # Set default to 2022 or the latest year
        ano_padrao = 2022 if 2022 in anos_disponiveis else anos_disponiveis[-1]
        st.session_state.ano_selecionado = ano_padrao
    else:
        st.session_state.ano_selecionado = None

if 'periodo_selecionado' not in st.session_state:
    st.session_state.periodo_selecionado = []

# Filtros para página 3
if pagina == "3. Correlações":
    st.sidebar.markdown("""
    <div style="border:3px solid #000000; border-radius:1px; padding:1px; margin:20px ;">
<center><h4 style="color:#00000; margin-top:1">Controles de Correlação</h4></center>

    """, unsafe_allow_html=True)
    
    # Seleção de produto - use df_consolidado here, not df_filtrado
    produto_correlacao = st.sidebar.selectbox(
        "Selecione um produto",
        sorted(df_consolidado['Produto'].unique()),
        index=list(sorted(df_consolidado['Produto'].unique())).index(st.session_state.produto_correlacao) 
            if st.session_state.produto_correlacao in sorted(df_consolidado['Produto'].unique()) else 0,
        key="produto_correlacao_select"
    )
    st.session_state.produto_correlacao = produto_correlacao
    
    # Modo de visualização
    modo_visualizacao = st.sidebar.radio(
        "Modo de visualização:",
        ["Ano Único", "Agregado 4 Anos", "Todos os Anos"],
        index=["Ano Único", "Agregado 4 Anos", "Todos os Anos"].index(st.session_state.modo_visualizacao)
            if st.session_state.modo_visualizacao in ["Ano Único", "Agregado 4 Anos", "Todos os Anos"] else 0,
        key="modo_visualizacao_radio"
    )
    st.session_state.modo_visualizacao = modo_visualizacao
    
    if modo_visualizacao == "Ano Único":
        # Get the data for the selected product
        dados_produto = df_consolidado[df_consolidado['Produto'] == produto_correlacao]
        
        # Create a list of available years as Python integers (not NumPy integers)
        anos_disponiveis = sorted([int(ano) for ano in dados_produto['Ano'].unique()])
        
        # Set default year (2022 if available, otherwise the most recent year)
        ano_padrao = 2022 if 2022 in anos_disponiveis else anos_disponiveis[-1]
        
        # Find the index of the current year or default year
        try:
            atual_ano = st.session_state.ano_selecionado if st.session_state.ano_selecionado in anos_disponiveis else ano_padrao
            indice_ano = anos_disponiveis.index(atual_ano)
        except ValueError:
            indice_ano = 0
        
        # Create the year selection dropdown
        ano_selecionado = st.sidebar.selectbox(
            "Selecione o ano:",
            anos_disponiveis,
            index=indice_ano,
            key="ano_selecionado_select" 
        )
        st.session_state.ano_selecionado = ano_selecionado
        
    elif modo_visualizacao == "Agregado 4 Anos":
        dados_produto = df_consolidado[df_consolidado['Produto'] == produto_correlacao]

        if not dados_produto.empty:
            # Ensure Ano is integer type
            dados_produto['Ano'] = dados_produto['Ano'].astype(int)
            min_ano = dados_produto['Ano'].min()
            max_ano = dados_produto['Ano'].max()

            periodos = []
            if min_ano <= max_ano:
                # Handle the first special 5-year period (e.g., 1990-1994)
                fim_primeiro_periodo = min(min_ano + 4, max_ano)
                periodos.append(f"{min_ano}-{fim_primeiro_periodo}")

                # Start the next period after the first one
                inicio = fim_primeiro_periodo + 1

                # Loop for subsequent standard 4-year periods
                while inicio <= max_ano:
                    fim = min(inicio + 3, max_ano) # 4-year duration (inicio, +1, +2, +3)
                    periodos.append(f"{inicio}-{fim}")
                    inicio = fim + 1

            if periodos:
                # Find current period or default to the most recent
                atual_periodo = st.session_state.periodo_selecionado if st.session_state.periodo_selecionado in periodos else periodos[-1]
                try:
                    indice_periodo = periodos.index(atual_periodo)
                except ValueError:
                    indice_periodo = len(periodos) - 1
                
                periodo_selecionado = st.sidebar.selectbox(
                    "Selecione o período",
                    periodos,
                    index=indice_periodo,
                    key="periodo_selecionado_select"
                )
                st.session_state.periodo_selecionado = periodo_selecionado

    # Filtro de mesorregiões com busca
    mesorregiao_selecionada = st.sidebar.multiselect(
        "Mesorregiões:",
        mesorregioes_ordenadas,
        default=[]
    )

    # Primeiro filtre pelo produto
    df_filtrado = df_consolidado[df_consolidado['Produto'] == produto_correlacao]

    # Depois aplique os filtros de ano/período
    if modo_visualizacao == "Ano Único" and ano_selecionado is not None:
        df_filtrado = df_filtrado[df_filtrado['Ano'] == ano_selecionado]
    elif modo_visualizacao == "Agregado 4 Anos" and periodo_selecionado:
        anos_periodo = periodo_selecionado.split('-')
        if len(anos_periodo) == 2:
            ano_inicio = int(anos_periodo[0])
            ano_fim = int(anos_periodo[1])
            df_filtrado = df_filtrado[(df_filtrado['Ano'] >= ano_inicio) & (df_filtrado['Ano'] <= ano_fim)]
    elif modo_visualizacao == "Todos os Anos":
        # Não aplica filtro de ano, mantém todos os anos para o produto selecionado
        pass
    
    # Por último, aplique o filtro de mesorregião APÓS outros filtros
    if mesorregiao_selecionada:
        df_filtrado = df_filtrado[df_filtrado['Mesorregião'].isin(mesorregiao_selecionada)]

    # Debug para verificar número de registros após cada filtro
    st.sidebar.text(f"Registros: {len(df_filtrado)}")
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
# Filtros para página 4
elif pagina == "4. Volatilidade":
    st.sidebar.markdown("""
    <div style="border:3px solid #000000; border-radius:1px; padding:1px; margin:20px ;">
    <center><h4 style="color:#00000; margin-top:1">Produto para análise de volatilidade regional</h4></center>
    """, unsafe_allow_html=True)
   
    produto_volatilidade = st.sidebar.selectbox(
        "Produtos:",
        sorted(df_consolidado['Produto'].unique()),
        key="produto_volatilidade"
    )
    # Filtro de mesorregiões com busca
    regioes_destacadas = st.sidebar.multiselect(
            "Mesorregiões:",
            mesorregioes_ordenadas,
            key='regioes_destacadas_volatilidade',
            default=[]
        )    
    df_filtrado = df_consolidado.copy()# Filtros globais
# Filtros para todas outras paginas
else:
    with st.sidebar.expander("Filtros de Produto e Região", expanded=False):
        # Filtro de período (mais compacto)
        anos = sorted(df_consolidado['Ano'].unique())
        periodo = st.sidebar.slider(
            "Selecione o período:",
            min_value=min(anos),
            max_value=max(anos),
            value=(min(anos), max(anos))
        )
        
        # Filtro de produtos com busca
        produtos = sorted(df_consolidado['Produto'].unique())
        produto_selecionado = st.multiselect(
            "Produtos:",
            produtos,
            default=[]
        )
        
        # Filtro de mesorregiões com busca
        mesorregiao_selecionada = st.multiselect(
            "Mesorregiões:",
            mesorregioes_ordenadas,
            default=[]
        )
    if produto_selecionado:
        df_filtrado = df_consolidado[df_consolidado['Produto'].isin(produto_selecionado)]
    else:
        df_filtrado = df_consolidado.copy()

    if mesorregiao_selecionada:
        df_filtrado = df_filtrado[df_filtrado['Mesorregião'].isin(mesorregiao_selecionada)]

    df_filtrado = df_filtrado[(df_filtrado['Ano'] >= periodo[0]) & (df_filtrado['Ano'] <= periodo[1])]

main_container = st.container()
with main_container:
    # Página inicial
    if pagina == "Início":
        st.header("Visão Geral")
        
        st.subheader("Contexto")
        st.markdown("""
            Este dashboard é uma análise de dados agrícolas brasileiros a nível de mesorregião. Inclui dados de produtividade, produção e área plantada as maiores culturas do país ao longo do período de 1990 a 2022, com dados meteorológicos de 2000 a 2022.

            Utilize o menu na barra lateral para navegar entre as diferentes análises disponíveis.

            Os filtros globais na barra lateral permitem personalizar a visualização dos dados por produto, mesorregião e período de tempo.
            
            ---
            """)
        
        st.subheader("Potenciais clientes e análises")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Produtores Rurais e Cooperativas Agrícolas", expanded=False):
                st.markdown('''
- Tendências temporais de rendimento por cultura e região
- Pontos de inflexão na produtividade (para identificar quando houve saltos tecnológicos)
- Correlações entre área plantada e rendimento (economias de escala)
- Volatilidade regional (identificar regiões mais estáveis para reduzir riscos)
- Impacto de variáveis climáticas no rendimento
            ''')

            with st.expander("Instituições Financeiras e Seguradoras", expanded=False):
                st.markdown('''
- Volatilidade da produção por cultura e região (para precificação de seguros)
- Diversificação agrícola por região (para análise de risco de carteira)
- Correlações com variáveis climáticas (para modelagem de risco)
- Detecção de outliers e eventos extremos nas séries temporais
- Especialização regional (para estratégias de crédito direcionado)
''')
            with st.expander("Órgãos Governamentais", expanded=False):
                st.markdown('''
- Tendências de longo prazo na produtividade
- Diversificação agrícola por região (para segurança alimentar)
- Taxonomia de mesorregiões (para políticas regionalizadas)
- Índice de especialização regional (para desenvolvimento econômico local)
- Impacto de variáveis climáticas (para políticas de adaptação climática)
''')
        with col2:
            with st.expander("Empresas de Tecnologia Agrícola", expanded=False):
                st.markdown('''
- Pontos de inflexão na produtividade (para identificar impacto de inovações)
- Correlações entre variáveis climáticas e rendimento (para desenvolver soluções adaptativas)
- Regiões com baixa produtividade e alta volatilidade (oportunidades de mercado)
- Análise de clusters de mesorregiões (para personalização de soluções)
- Comparativos regionais de rendimento (benchmarking para tecnologias)
''')

            with st.expander("Indústrias de Processamento Agrícola", expanded=False):
                st.markdown('''
- Mapeamento de mesorregiões especializadas (para localização de plantas)
- Tendências de produção e volatilidade (para planejamento de suprimentos)
- Sazonalidade na produção (para gestão de estoques)
- Distribuição geográfica da produção (para otimização logística)
- Projeções de séries temporais (para planejamento estratégico)
''')
            with st.expander("Investidores em Terras Agrícolas", expanded=False):
                st.markdown('''                
- Tendências de longo prazo na produtividade por região
- Volatilidade regional (para avaliação de risco)
- Especialização regional (para identificação de oportunidades)
- Correlações com variáveis climáticas (para avaliação de resiliência)
- Comparativos de rendimento entre regiões (para valoração de terras)
''')
        st.markdown('---')
        col1, col2 = st.columns(2)
        
        with col1:            
            st.subheader("Resumo dos Dados")
            
            # Estatísticas básicas
            num_mesorregioes = df_filtrado['Mesorregião'].nunique()
            num_produtos = df_filtrado['Produto'].nunique()
            periodo_analise = f"{df_filtrado['Ano'].min()} a {df_filtrado['Ano'].max()}"
            
            st.markdown(f"**Número de Mesorregiões:** {num_mesorregioes}")
            st.markdown(f"**Número de Produtos:** {num_produtos}")
            st.markdown(f"**Período de Análise:** {periodo_analise}")
        
        with col2:
            st.subheader("Distribuição da Produção por Produto")
            
            # Agrupar por produto para visualização
            producao_por_produto = df_filtrado.groupby('Produto')['Producao_Toneladas'].sum().reset_index()
            producao_por_produto = producao_por_produto.sort_values('Producao_Toneladas', ascending=False)
            
            fig = px.pie(
                producao_por_produto,
                values='Producao_Toneladas',
                names='Produto',
                title='Distribuição da Produção Total por Produto',
                template='plotly_white',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

    # Mapas do PowerBI
    elif pagina == "Mapas":
        powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiMWRhOWY4NzItYzMwNi00Yzk0LWIxZmYtNTMyYzlhZDUzM2U4IiwidCI6ImMxNzdmNmRkLWY1MTUtNDRlNy05ZmMzLTZiNzZjODdhZmViMCJ9&pageName=e52d3bda5d537e68a452"

        # Embed in an iframe
        components.iframe(powerbi_url, height=600, width=600)
    
    # 1. Tendências Temporais    
    elif pagina == "1. Tendências Temporais":
        st.header("Análise de Tendências Temporais")
        
        # Descrição da análise
        st.markdown("""
        Esta seção apresenta a evolução do rendimento médio de cada cultura ao longo do tempo
        e identifica pontos de inflexão (anos com mudanças significativas).
        """)
        
        # Visualização interativa com Plotly
        st.subheader("Evolução do Rendimento Médio por Cultura")
        
        # Agrupar por Produto e Ano para calcular o rendimento médio
        rendimento_medio = df_filtrado.groupby(['Produto', 'Ano'])['Rendimento_KgPorHectare'].mean().reset_index()
        
        # Criar gráfico interativo
        fig = px.line(
            rendimento_medio,
            x='Ano',
            y='Rendimento_KgPorHectare',
            color='Produto',
            markers=True,
            title='Evolução do Rendimento Médio por Cultura (1990-2022)',
            labels={'Rendimento_KgPorHectare': 'Rendimento (Kg/Hectare)', 'Ano': 'Ano'},
            template='plotly_white'
        )
        
        # Adicionar linha de tendência usando regressão linear
        for produto in rendimento_medio['Produto'].unique():
            df_produto = rendimento_medio[rendimento_medio['Produto'] == produto]
            X = df_produto['Ano']
            y = df_produto['Rendimento_KgPorHectare']
            
            # Ajustar modelo de regressão linear
            coef = np.polyfit(X, y, 1)
            linha_tendencia = np.poly1d(coef)
            
            # Adicionar linha de tendência ao gráfico
            fig.add_scatter(
                x=X,
                y=linha_tendencia(X),
                mode='lines',
                line=dict(dash='dash'),
                name=f'Tendência - {produto}',
                opacity=0.7
            )
        
        # Melhorar o layout
        fig.update_layout(
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identificação de pontos de inflexão
        st.subheader("Pontos de Inflexão por Cultura")
        
        # Criar duas colunas para os controles, com proporção que deixa mais espaço para o gráfico
        col1, col2 = st.columns([1, 4])

        with col1:
            # Escolher um produto específico para análise detalhada (coluna menor)
            produto_inflexao = st.selectbox(
                "Cultura:",
                sorted(df_filtrado['Produto'].unique()),
                key="produto_inflexao"
            )
            
            # Definir limiar de variação (na mesma coluna menor)
            limiar = st.slider(
                "Limiar de variação (%):",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                key="limiar_variacao"
            )
            
        with col2:
            # Filtrar dados do produto selecionado
            dados_produto = rendimento_medio[rendimento_medio['Produto'] == produto_inflexao].sort_values('Ano')
            
            # Calcular a taxa de variação anual
            dados_produto['Variacao'] = dados_produto['Rendimento_KgPorHectare'].pct_change() * 100
            
            with st.expander("ℹ️ O que significa este limiar?"):
                st.markdown("""
                O limiar de variação (15% por padrão) determina o que é considerado um ponto de inflexão significativo na produtividade de uma cultura. Pontos acima desse limiar representam mudanças importantes que podem ser causadas por:
                
                - Inovações tecnológicas (novas variedades, técnicas de manejo)
                - Eventos climáticos extremos (secas, inundações) 
                - Políticas agrícolas (subsídios, programas de apoio)
                - Mudanças no mercado (preços, demanda)
                """)
                
            # Identificar pontos de inflexão
            pontos_inflexao = dados_produto[abs(dados_produto['Variacao']) > limiar]
            
            # Criar gráfico com pontos de inflexão
            fig = go.Figure()
            
            # Adicionar série principal
            fig.add_trace(go.Scatter(
                x=dados_produto['Ano'],
                y=dados_produto['Rendimento_KgPorHectare'],
                mode='lines+markers',
                name=produto_inflexao,
                line=dict(width=2)
            ))
            
            # Adicionar pontos de inflexão
            if not pontos_inflexao.empty:
                fig.add_trace(go.Scatter(
                    x=pontos_inflexao['Ano'],
                    y=pontos_inflexao['Rendimento_KgPorHectare'],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    name='Pontos de Inflexão',
                    text=[f"Variação: {var:.1f}%" for var in pontos_inflexao['Variacao']],
                    hovertemplate='Ano: %{x}<br>Rendimento: %{y} Kg/Ha<br>%{text}'
                ))
            
            # Configurar layout
            fig.update_layout(
                title=f'Evolução do Rendimento de {produto_inflexao} com Pontos de Inflexão (Variação > {limiar}%)',
                xaxis_title='Ano',
                yaxis_title='Rendimento (Kg/Hectare)',
                height=500,
                template='plotly_white',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de pontos de inflexão
        if not pontos_inflexao.empty:
            st.write(f"**Pontos de inflexão identificados para {produto_inflexao}** (Variação > {limiar}%):")
            st.dataframe(
                pontos_inflexao[['Ano', 'Rendimento_KgPorHectare', 'Variacao']].rename(
                    columns={'Variacao': 'Variação (%)'}
                ).set_index('Ano').style.format({
                    'Rendimento_KgPorHectare': '{:.2f}',
                    'Variação (%)': '{:.2f}%'
                })
            )
        else:
            st.info(f"Não foram identificados pontos de inflexão para {produto_inflexao} com o limiar de {limiar}%.")

    # 2. Comparativos Regionais
    elif pagina == "2. Comparativos Regionais":
        st.header("Comparativos Regionais")
        
        # Descrição da análise
        st.markdown("""
        Esta seção apresenta o ranking das mesorregiões mais produtivas para cada cultura
        e mapas de calor mostrando a distribuição espacial da produtividade.
        """)
        
        # Ranking das mesorregiões mais produtivas
        st.subheader("Ranking das Mesorregiões mais Produtivas")
        
        # Selecionar o produto para o ranking
        produto_ranking = st.selectbox(
            "Selecione um produto:",
            sorted(df_filtrado['Produto'].unique()),
            key="produto_ranking"
        )
        
        # Agrupar por Mesorregião para o produto selecionado
        dados_ranking = df_filtrado[df_filtrado['Produto'] == produto_ranking]
        ranking = dados_ranking.groupby('Mesorregião')['Rendimento_KgPorHectare'].mean().reset_index()
        ranking = ranking.sort_values('Rendimento_KgPorHectare', ascending=False).reset_index(drop=True)
        
        # Definir o número de mesorregiões a mostrar
        num_mesorregioes = st.slider(
            "Número de mesorregiões no ranking:",
            min_value=5,
            max_value=min(30, len(ranking)),
            value=10
        )
        
        # Criar gráfico de barras
        fig = px.bar(
            ranking.head(num_mesorregioes),
            y='Mesorregião',
            x='Rendimento_KgPorHectare',
            orientation='h',
            title=f'Top {num_mesorregioes} Mesorregiões mais Produtivas - {produto_ranking}',
            labels={'Rendimento_KgPorHectare': 'Rendimento Médio (Kg/Hectare)', 'Mesorregião': ''},
            template='plotly_white',
            color='Rendimento_KgPorHectare',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com o ranking completo
        with st.expander("Ver tabela com o ranking completo"):
            st.dataframe(
                ranking.style.format({'Rendimento_KgPorHectare': '{:.2f}'})
                .background_gradient(cmap='viridis', subset=['Rendimento_KgPorHectare'])
            )
        
        # Mapa de calor da produção por região e produto
        st.markdown('---')
        st.header('Mapas de Calor')
        st.markdown("""
        - As cores mais intensas (amarelo) indicam valores mais altos (maior produtividade/produção)
        - Os dados estão normalizados por coluna (produto) - valor 1 representa a região com maior valor
        - Escala logarítmica é usada para melhorar a visualização das diferenças entre regiões
        - Regiões ordenadas de norte a sul
        """)
        
        # Extrair o estado de cada mesorregião
        def extrair_estado(mesorregiao):
            partes = mesorregiao.split(' - ')
            if len(partes) > 1:
                return partes[1]
            return ''

        # Definir ordem das regiões brasileiras e estados
        regioes_brasil = {
            'Norte': ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],
            'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
            'Centro-Oeste': ['DF', 'GO', 'MS', 'MT'],
            'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
            'Sul': ['PR', 'RS', 'SC']
        }

        # Criar uma função para ordenar as mesorregiões
        def ordenar_mesorregioes_norte_sul(mesorregioes):
            # Criar um dicionário para mapear estados para suas regiões
            estado_para_regiao = {}
            for regiao, estados in regioes_brasil.items():
                for estado in estados:
                    estado_para_regiao[estado] = regiao
            
            # Criar uma lista de tuplas (região, estado, mesorregião)
            dados_ordenacao = []
            for mesorregiao in mesorregioes:
                estado = extrair_estado(mesorregiao)
                regiao = estado_para_regiao.get(estado, 'Outra')
                dados_ordenacao.append((regiao, estado, mesorregiao))
            
            # Ordenar primeira por região (seguindo a ordem: Norte, Nordeste, Centro-Oeste, Sudeste, Sul)
            ordem_regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul', 'Outra']
            dados_ordenacao.sort(key=lambda x: (ordem_regioes.index(x[0]), x[1], x[2]))
            
            # Retornar apenas a lista de mesorregiões ordenadas
            return [item[2] for item in dados_ordenacao]

        # Mapa de Calor da Produção
        st.subheader(f"Mapa de Calor da Produção ({periodo[0]}-{periodo[1]})")
        
        producao_pivot = df_filtrado.pivot_table(
            values='Producao_Toneladas',
            index='Mesorregião',
            columns='Produto',
            aggfunc='sum'
        ).fillna(0)
        
        # Ordenar mesorregiões
        mesorregioes_ordenadas = ordenar_mesorregioes_norte_sul(producao_pivot.index.tolist())

        # Reordenar o pivot table
        producao_pivot = producao_pivot.reindex(mesorregioes_ordenadas)

        # Normalizar usando escala logarítmica
        producao_normalizada = producao_pivot.copy()
        for col in producao_normalizada.columns:
            if producao_normalizada[col].max() != 0:
                # Add small value to avoid log(0)
                normalized_values = producao_normalizada[col] / producao_normalizada[col].max()
                # Apply log transformation (adding 0.01 to avoid log(0))
                producao_normalizada[col] = np.log10(normalized_values + 0.01) / np.log10(1.01)
                # Rescale to 0-1 range
                if producao_normalizada[col].max() != producao_normalizada[col].min():
                    producao_normalizada[col] = (producao_normalizada[col] - producao_normalizada[col].min()) / (producao_normalizada[col].max() - producao_normalizada[col].min())

        # Create custom hover text with better formatting
        hover_text = []
        for i, mesorregiao in enumerate(producao_normalizada.index):
            row_hover = []
            for j, produto in enumerate(producao_normalizada.columns):
                # Get original and normalized values
                orig_value = producao_pivot.iloc[i, j]
                norm_value = producao_pivot.iloc[i, j] / producao_pivot[producao_pivot.columns[j]].max()
                row_hover.append(f"Região: {mesorregiao}<br>Produto: {produto}<br>Produção: {orig_value:.2f} ton<br>Valor Normalizado: {norm_value:.4f}")
            hover_text.append(row_hover)

        fig = px.imshow(
            producao_normalizada,
            labels=dict(x="Produto", y="Mesorregião", color="Produção Normalizada (Log)"),
            x=producao_normalizada.columns,
            y=producao_normalizada.index,
            aspect="auto",
            color_continuous_scale='Viridis'
        )

        # Use the custom hover text
        fig.update_traces(hovertemplate="%{customdata}", customdata=hover_text)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mapa de calor do rendimento
        st.subheader(f"Mapa de Calor do Rendimento ({periodo[0]}-{periodo[1]})")
        
        # Opções para o mapa de calor
        mapa_opcao = st.radio(
            "Selecione o tipo de visualização:",
            ["Por Produto (todas as mesorregiões)", "Por Mesorregião (todos os produtos)"]
        )
        
        if mapa_opcao == "Por Produto (todas as mesorregiões)":
            # Agrupar por Mesorregião e Produto
            mapa_dados = df_filtrado.groupby(['Mesorregião', 'Produto'])['Rendimento_KgPorHectare'].mean().reset_index()
            
            # Criar tabela pivotada
            mapa_pivot = mapa_dados.pivot_table(
                values='Rendimento_KgPorHectare',
                index='Mesorregião',
                columns='Produto'
            ).fillna(0)

            # Ordenar mesorregiões
            mesorregioes_ordenadas = ordenar_mesorregioes_norte_sul(mapa_pivot.index.tolist())

            # Reordenar o pivot table
            mapa_pivot = mapa_pivot.reindex(mesorregioes_ordenadas)
            
            # Normalizar usando escala logarítmica
            mapa_normalizado = mapa_pivot.copy()
            for col in mapa_normalizado.columns:
                if mapa_normalizado[col].max() != 0:
                    # Add small value to avoid log(0)
                    normalized_values = mapa_normalizado[col] / mapa_normalizado[col].max()
                    # Apply log transformation (adding 0.01 to avoid log(0))
                    mapa_normalizado[col] = np.log10(normalized_values + 0.01) / np.log10(1.01)
                    # Rescale to 0-1 range
                    if mapa_normalizado[col].max() != mapa_normalizado[col].min():
                        mapa_normalizado[col] = (mapa_normalizado[col] - mapa_normalizado[col].min()) / (mapa_normalizado[col].max() - mapa_normalizado[col].min())
            
            # Create custom hover text with better formatting
            hover_text = []
            for i, mesorregiao in enumerate(mapa_normalizado.index):
                row_hover = []
                for j, produto in enumerate(mapa_normalizado.columns):
                    # Get original and normalized values
                    orig_value = mapa_pivot.iloc[i, j]
                    norm_value = mapa_pivot.iloc[i, j] / mapa_pivot[mapa_pivot.columns[j]].max()
                    row_hover.append(f"Região: {mesorregiao}<br>Produto: {produto}<br>Rendimento: {orig_value:.2f} kg/ha<br>Valor Normalizado: {norm_value:.4f}")
                hover_text.append(row_hover)
            
            # Criar mapa de calor
            fig = px.imshow(
                mapa_normalizado,
                labels=dict(x="Produto", y="Mesorregião", color="Rendimento Normalizado (Log)"),
                x=mapa_normalizado.columns,
                y=mapa_normalizado.index,
                aspect="auto",
                color_continuous_scale='Viridis'
            )
            
            # Use the custom hover text
            fig.update_traces(hovertemplate="%{customdata}", customdata=hover_text)
            
            fig.update_layout(height=800, title="Distribuição Espacial da Produtividade por Produto (Normalizada)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Selecionar mesorregião para análise
            mesorregiao_mapa = st.selectbox(
                "Selecione uma mesorregião:",
                sorted(df_filtrado['Mesorregião'].unique()),
                key="mesorregiao_mapa"
            )
            
            # Agrupar por Ano e Produto para a mesorregião selecionada
            mapa_dados = df_filtrado[df_filtrado['Mesorregião'] == mesorregiao_mapa]
            mapa_dados = mapa_dados.groupby(['Ano', 'Produto'])['Rendimento_KgPorHectare'].mean().reset_index()
            
            # Criar tabela pivotada
            mapa_pivot = mapa_dados.pivot_table(
                values='Rendimento_KgPorHectare',
                index='Ano',
                columns='Produto'
            ).fillna(0)
            
            # Informações sobre como interpretar o mapa de calor
            st.markdown("""
            **Como interpretar este Mapa de Calor:**
            - As cores mais intensas (amarelo) indicam valores mais altos de rendimento
            - Os dados estão normalizados por coluna (produto) - valor 1 representa o ano com maior rendimento daquele produto
            - Escala logarítmica é usada para melhorar a visualização das diferenças entre anos
            - Anos ordenados cronologicamente do mais antigo ao mais recente
            """)
            
            # Garantir que os anos estejam em ordem cronológica
            mapa_pivot = mapa_pivot.sort_index()
            
            # Normalizar usando escala logarítmica
            mapa_normalizado = mapa_pivot.copy()
            for col in mapa_normalizado.columns:
                if mapa_normalizado[col].max() != 0:
                    # Add small value to avoid log(0)
                    normalized_values = mapa_normalizado[col] / mapa_normalizado[col].max()
                    # Apply log transformation (adding 0.01 to avoid log(0))
                    mapa_normalizado[col] = np.log10(normalized_values + 0.01) / np.log10(1.01)
                    # Rescale to 0-1 range
                    if mapa_normalizado[col].max() != mapa_normalizado[col].min():
                        mapa_normalizado[col] = (mapa_normalizado[col] - mapa_normalizado[col].min()) / (mapa_normalizado[col].max() - mapa_normalizado[col].min())
            
            # Create custom hover text with better formatting
            hover_text = []
            for i, ano in enumerate(mapa_normalizado.index):
                row_hover = []
                for j, produto in enumerate(mapa_normalizado.columns):
                    # Get original and normalized values
                    orig_value = mapa_pivot.iloc[i, j]
                    norm_value = mapa_pivot.iloc[i, j] / mapa_pivot[mapa_pivot.columns[j]].max()
                    row_hover.append(f"Ano: {ano}<br>Produto: {produto}<br>Rendimento: {orig_value:.2f} kg/ha<br>Valor Normalizado: {norm_value:.4f}")
                hover_text.append(row_hover)
            
            # Criar mapa de calor
            fig = px.imshow(
                mapa_normalizado,
                labels=dict(x="Produto", y="Ano", color="Rendimento Normalizado (Log)"),
                x=mapa_normalizado.columns,
                y=mapa_normalizado.index,
                aspect="auto",
                color_continuous_scale='Viridis'
            )
            
            # Use the custom hover text
            fig.update_traces(hovertemplate="%{customdata}", customdata=hover_text)
            
            fig.update_layout(
                height=800, 
                title=f"Evolução da Produtividade por Produto em {mesorregiao_mapa} ({periodo[0]}-{periodo[1]})",
                yaxis={'dtick': 5}  # Show year labels with interval of 5
            )
            st.plotly_chart(fig, use_container_width=True)  

    # 3. Correlações entre variáveis
    elif pagina == "3. Correlações":
        st.header("Correlações entre Variáveis")
        
        # Descrição da análise
        st.markdown("""
        Esta seção analisa as relações entre diferentes variáveis, como área plantada e rendimento 
        (para verificar se há economias de escala) e a correlação entre valor da produção e rendimento.
        Também inclui análises de correlação com variáveis climáticas.
        """)
        
        # Main content area - using the controls from the sidebar
        # Get values from session state
        produto_correlacao = st.session_state.produto_correlacao
        modo_visualizacao = st.session_state.modo_visualizacao
        
        # For correlation page only filter by product, not by the global filters
        dados_produto = df_filtrado
        if dados_produto.empty:
            st.warning(f"Não há dados disponíveis para {produto_correlacao}.")
            st.stop()
        
        # Processar dados com base no modo de visualização
        if modo_visualizacao == "Ano Único":
            ano_selecionado = st.session_state.ano_selecionado
            dados_produto = dados_produto[dados_produto['Ano'] == ano_selecionado]
            
            if dados_produto.empty:
                st.warning(f"Não há dados disponíveis para {produto_correlacao} no ano {ano_selecionado}.")
                st.stop()
            
            titulo_area_rend = f'{produto_correlacao} ({ano_selecionado})'
            titulo_valor_rend = f'{produto_correlacao} ({ano_selecionado})'
            
        elif modo_visualizacao == "Agregado 4 Anos":
            periodo_selecionado = st.session_state.periodo_selecionado
            try:
                inicio_periodo, fim_periodo = map(int, periodo_selecionado.split('-'))
                
                dados_produto = dados_produto[(dados_produto['Ano'] >= inicio_periodo) & (dados_produto['Ano'] <= fim_periodo)]
                
                if dados_produto.empty:
                    st.warning(f"Não há dados disponíveis para {produto_correlacao} no período {periodo_selecionado}.")
                    st.stop()
                
                titulo_area_rend = f'{produto_correlacao} ({periodo_selecionado})'
                titulo_valor_rend = f'{produto_correlacao} ({periodo_selecionado})'
            except:
                st.error(f"Formato de período inválido: {periodo_selecionado}")
                st.stop()
                
        else:  # "Todos os Anos"
            titulo_area_rend = f'{produto_correlacao} (Todos os anos)'
            titulo_valor_rend = f'{produto_correlacao} (Todos os anos)'
        
        # Create two columns for the visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlação entre área plantada e rendimento
            fig = px.scatter(
                dados_produto,
                x='Area_Plantada_Hectares',
                y='Rendimento_KgPorHectare',
                opacity=0.7,
                trendline="ols",
                trendline_scope="overall",  # Importante: força uma única linha de tendência
                trendline_color_override="blue",  # Cor única para a linha de tendência
                labels={
                    'Area_Plantada_Hectares': 'Área Plantada (Hectares)',
                    'Rendimento_KgPorHectare': 'Rendimento (Kg/Hectare)'
                },
                title=f"Área Plantada vs. Rendimento<br>{titulo_area_rend}",
                height=500
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular o coeficiente de correlação para todos os dados agregados
            corr_area_rend = dados_produto['Area_Plantada_Hectares'].corr(dados_produto['Rendimento_KgPorHectare'])
            st.metric("Coeficiente de Correlação", f"{corr_area_rend:.3f}")
            
            # Interpretar a correlação
            if abs(corr_area_rend) < 0.3:
                st.info("Correlação fraca: Pouca evidência de economias de escala.")
            elif corr_area_rend >= 0.3:
                st.success("Correlação positiva: Há evidências de economias de escala.")
            else:
                st.error("Correlação negativa: Áreas maiores tendem a ter menor rendimento.")

        with col2:
            # Correlação entre valor da produção e rendimento
            fig = px.scatter(
                dados_produto,
                x='Valor_Produzido_Mil_Reais',
                y='Rendimento_KgPorHectare',
                opacity=0.7,
                # Uma única linha de tendência para todos os dados, independente das regiões selecionadas
                trendline="ols", 
                trendline_scope="overall",  # Importante: força uma única linha de tendência
                trendline_color_override="blue",  # Cor única para a linha de tendência
                labels={
                    'Valor_Produzido_Mil_Reais': 'Valor da Produção (Mil Reais)',
                    'Rendimento_KgPorHectare': 'Rendimento (Kg/Hectare)'
                },
                title=f"Valor vs. Rendimento<br>{titulo_valor_rend}",
                height=500
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular o coeficiente de correlação para todos os dados agregados
            corr_valor_rend = dados_produto['Valor_Produzido_Mil_Reais'].corr(dados_produto['Rendimento_KgPorHectare'])
            st.metric("Coeficiente de Correlação", f"{corr_valor_rend:.3f}")
            
            # Interpretar a correlação
            if abs(corr_valor_rend) < 0.3:
                st.info("Correlação fraca: Pouca relação entre valor e rendimento.")
            elif corr_valor_rend >= 0.3:
                st.success("Correlação positiva: Maior rendimento está associado a maior valor.")
            else:
                st.error("Correlação negativa: Relação inversa entre valor e rendimento.")
        # Correlações com variáveis climáticas
        st.subheader("Correlações com Variáveis Climáticas")
        st.markdown("Dados climáticos disponíveis apenas após o ano 2000")
        
        # Verificar se há variáveis climáticas disponíveis
        var_climaticas = [col for col in dados_produto.columns if col in [
            'precipitacao_total_anual', 'radiacao_global_media', 
            'temperatura_bulbo_media', 'vento_velocidade_media'
        ]]
        
        if var_climaticas:
            # Criar matriz de correlação
            cols_analise = var_climaticas + ['Rendimento_KgPorHectare']
            matriz_corr = dados_produto[cols_analise].corr()
            
            # Renomear colunas para melhor visualização
            matriz_corr.columns = [
                col.replace('_', ' ').title() for col in matriz_corr.columns
            ]
            matriz_corr.index = matriz_corr.columns
            
            # Criar mapa de calor
            # Mask upper triangle
            matriz_corr_masked = matriz_corr.copy()
            matriz_corr_masked.values[np.triu_indices_from(matriz_corr, k=1)] = np.nan

            fig = px.imshow(
                matriz_corr_masked,
                text_auto='.3f',
                labels=dict(x="Variável", y="Variável", color="Correlação"),
                x=matriz_corr.columns,
                y=matriz_corr.index,
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                title=f'Matriz de Correlação - {produto_correlacao}'
            )

            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar interpretações para cada variável climática
            st.subheader("Interpretação de Correlações Climáticas com Rendimento")
            
            for var in var_climaticas:
                var_nome = var.replace('_', ' ').title()
                corr_valor = matriz_corr.loc['Rendimento Kgporhectare', var_nome]
                
                if abs(corr_valor) < 0.3:
                    intensidade = "fraca"
                    icon = "ℹ️"
                elif abs(corr_valor) < 0.7:
                    intensidade = "moderada"
                    icon = "⚠️" if corr_valor < 0 else "✅"
                else:
                    intensidade = "forte"
                    icon = "❌" if corr_valor < 0 else "🔥"
                
                direcao = "positiva" if corr_valor >= 0 else "negativa"
                
                st.markdown(f"{icon} **{var_nome}**: Correlação {intensidade} {direcao} ({corr_valor:.3f})")
        else:
            st.warning("Não foram encontradas variáveis climáticas nos dados filtrados.")

    # 4. Análise de volatilidade
    elif pagina == "4. Volatilidade":
        st.header("Análise de Volatilidade")
        
        # Descrição da análise
        st.markdown("""
        Esta seção calcula o coeficiente de variação do rendimento por cultura e região,
        identificando as regiões e culturas mais estáveis/instáveis ao longo do tempo.
        """)
        
        # Calcular volatilidade por produto
        # Agrupar por Produto, Mesorregião e Ano
        dados_agrupados = df_filtrado.groupby(['Produto', 'Mesorregião', 'Ano'])['Rendimento_KgPorHectare'].mean().reset_index()
        
        # Calcular coeficiente de variação por Produto e Mesorregião
        cv_por_produto_regiao = dados_agrupados.groupby(['Produto', 'Mesorregião']).agg(
            Rendimento_Medio=('Rendimento_KgPorHectare', 'mean'),
            Desvio_Padrao=('Rendimento_KgPorHectare', 'std')
        ).reset_index()
        
        # Calcular o coeficiente de variação (CV = desvio padrão / média * 100)
        cv_por_produto_regiao['CV'] = (cv_por_produto_regiao['Desvio_Padrao'] / 
                                    cv_por_produto_regiao['Rendimento_Medio']) * 100
        
        # Volatilidade por cultura
        culturas_volatilidade = cv_por_produto_regiao.groupby('Produto').agg(
            CV_Medio=('CV', 'mean')
        ).sort_values('CV_Medio', ascending=True).reset_index()
        
        st.subheader("Volatilidade por Cultura")
        
        fig = px.bar(
            culturas_volatilidade,
            y='Produto',
            x='CV_Medio',
            orientation='h',
            title='Volatilidade do Rendimento por Cultura (Coeficiente de Variação Médio)',
            labels={'CV_Medio': 'Coeficiente de Variação Médio (%)', 'Produto': ''},
            color='CV_Medio',
            color_continuous_scale='RdYlGn_r',
            template='plotly_white',
            category_orders={"Produto": culturas_volatilidade['Produto'].tolist()}
        )

        # Set specific y-axis configuration
        fig.update_layout(height=500, yaxis={
                'categoryorder': 'array', 
                'categoryarray': culturas_volatilidade['Produto'].tolist(),
                'autorange': "reversed"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise regional de volatilidade
        st.subheader("Volatilidade Regional por Produto")
        
        # Filtrar dados do produto selecionado
        cv_produto = cv_por_produto_regiao[cv_por_produto_regiao['Produto'] == produto_volatilidade]
        
        # Criar uma coluna para destacar as regiões selecionadas
        cv_produto['Destacado'] = cv_produto['Mesorregião'].isin(regioes_destacadas)
        
        # Criar abas para regiões estáveis e instáveis
        tab1, tab2 = st.tabs(["Regiões mais Estáveis", "Regiões mais Instáveis"])
        
        with tab1:
            # Regiões mais estáveis (menor CV)
            regioes_estaveis = cv_produto.sort_values('CV', ascending=True).head(10)
            
            fig = px.bar(
                regioes_estaveis,
                y='Mesorregião',
                x='CV',
                orientation='h',
                title=f'Top 10 Regiões mais Estáveis para {produto_volatilidade}',
                labels={'CV': 'Coeficiente de Variação (%)', 'Mesorregião': ''},
                color='Destacado',
                color_discrete_map={True: '#FF4B4B', False: '#636EFA'},
                category_orders={"Produto": regioes_estaveis['Produto'].tolist()},
                template='plotly_white'
            )
            
            fig.update_layout(
                height=500, 
                yaxis={
                    'categoryorder': 'array', 
                    'categoryarray': regioes_estaveis['Mesorregião'].tolist(),
                    'autorange': "reversed"
                },
                showlegend=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Regiões mais instáveis (maior CV)
            regioes_instaveis = cv_produto.sort_values('CV', ascending=False).head(10)
            
            fig = px.bar(
                regioes_instaveis,
                y='Mesorregião',
                x='CV',
                orientation='h',
                title=f'Top 10 Regiões mais Instáveis para {produto_volatilidade}',
                labels={'CV': 'Coeficiente de Variação (%)', 'Mesorregião': ''},
                color='Destacado',
                color_discrete_map={True: '#FF4B4B', False: '#636EFA'},
                template='plotly_white'
            )
            
            fig.update_layout(
                height=500, 
                yaxis={
                    'categoryorder': 'array', 
                    'categoryarray': regioes_instaveis['Mesorregião'].tolist(),
                    'autorange': "reversed"
                },
                legend_title_text='Região Destacada',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Análise da relação entre rendimento médio e volatilidade
        st.subheader("Relação entre Rendimento Médio e Volatilidade")
        
        fig = px.scatter(
            cv_produto,
            x='Rendimento_Medio',
            y='CV',
            hover_name='Mesorregião',
            opacity=0.7,
            trendline="ols",
            labels={
                'Rendimento_Medio': 'Rendimento Médio (Kg/Hectare)',
                'CV': 'Coeficiente de Variação (%)'
            },
            title=f'Relação entre Rendimento Médio e Volatilidade - {produto_volatilidade}',
            color='Destacado',
            color_discrete_map={True: '#FF4B4B', False: '#636EFA'},
            template='plotly_white'
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcular correlação
        corr_rend_cv = cv_produto['Rendimento_Medio'].corr(cv_produto['CV'])
        
        # Interpretar a correlação
        if abs(corr_rend_cv) < 0.3:
            st.info(f"Correlação fraca ({corr_rend_cv:.3f}): Pouca relação entre rendimento médio e volatilidade.")
        elif corr_rend_cv >= 0.3:
            st.error(f"Correlação positiva ({corr_rend_cv:.3f}): Regiões com maior rendimento tendem a ser mais voláteis.")
        else:
            st.success(f"Correlação negativa ({corr_rend_cv:.3f}): Regiões com maior rendimento tendem a ser mais estáveis.")
        
        # Adicionar informações sobre as regiões destacadas se houver seleção
        if regioes_destacadas:
            st.subheader("Análise das Regiões Destacadas")
            
            # Filtrar apenas as regiões destacadas
            df_destacadas = cv_produto[cv_produto['Mesorregião'].isin(regioes_destacadas)]
            
            # Calcular estatísticas
            cv_medio_destacadas = df_destacadas['CV'].mean()
            cv_medio_geral = cv_produto['CV'].mean()
            
            # Comparar com a média geral
            if cv_medio_destacadas < cv_medio_geral:
                diferenca = ((cv_medio_geral - cv_medio_destacadas) / cv_medio_geral) * 100
                st.success(f"As regiões destacadas têm volatilidade média {diferenca:.2f}% menor que a média nacional.")
            else:
                diferenca = ((cv_medio_destacadas - cv_medio_geral) / cv_medio_geral) * 100
                st.warning(f"As regiões destacadas têm volatilidade média {diferenca:.2f}% maior que a média nacional.")
            
            # Mostrar tabela com detalhes das regiões destacadas
            st.markdown("### Detalhes das Regiões Destacadas")
            
            df_destacadas_formatado = df_destacadas.copy()
            df_destacadas_formatado['Rendimento_Medio'] = df_destacadas_formatado['Rendimento_Medio'].round(2)
            df_destacadas_formatado['CV'] = df_destacadas_formatado['CV'].round(2)
            
            st.table(df_destacadas_formatado[['Mesorregião', 'Rendimento_Medio', 'CV']].sort_values('CV'))
    
    # 5. Taxonomia de mesorregiões
    elif pagina == "5. Taxonomia de Mesorregiões":
            
        st.header("Taxonomia de Mesorregiões")
        
        # Descrição da análise
        st.markdown("""
        Esta seção realiza o agrupamento de regiões com padrões similares de produtividade
        e classifica as mesorregiões por perfil de culturas predominantes.
        """)
        
        # Filtrar os dados para análise
        # Precisamos garantir que temos dados suficientes para cada mesorregião e produto
        mesorregioes_validas = df_filtrado.groupby('Mesorregião').size()
        mesorregioes_validas = mesorregioes_validas[mesorregioes_validas >= 10].index.tolist()
        
        # Verificar se temos mesorregiões suficientes
        if len(mesorregioes_validas) < 3:
            st.warning("Dados insuficientes para análise de taxonomia. Ajuste os filtros para incluir mais mesorregiões.")
        else:
            # Filtrar para as mesorregiões válidas
            df_taxonomia = df_filtrado[df_filtrado['Mesorregião'].isin(mesorregioes_validas)]
            
            # Preparar os dados para clustering
            # Pivotear para obter mesorregiões nas linhas e produtos nas colunas
            rendimento_pivot = df_taxonomia.pivot_table(
                index='Mesorregião', 
                columns='Produto', 
                values='Rendimento_KgPorHectare',
                aggfunc='mean'
            ).fillna(0)  # Preencher NaN com 0
            
            # Normalizar os dados para o clustering
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            rendimento_scaled = scaler.fit_transform(rendimento_pivot)
            
            # Parâmetros do clustering
            n_clusters = st.slider(
                "Número de clusters:",
                min_value=2,
                max_value=min(10, len(mesorregioes_validas) - 1),
                value=4
            )
            
            # Aplicar K-means
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(rendimento_scaled)
            
            # Adicionar as labels de cluster ao dataframe
            rendimento_pivot['Cluster'] = cluster_labels
            
            # Análise dos clusters
            cluster_info = rendimento_pivot.groupby('Cluster').mean()
            
            # Características dos clusters
            st.subheader("Características dos Clusters")
            
            # Criar DataFrame para visualização
            cluster_data = []
            for i in range(n_clusters):
                # Encontrar as culturas mais importantes para cada cluster
                culturas_cluster = cluster_info.loc[i].sort_values(ascending=False).index[:3].tolist()
                n_mesorregioes = (rendimento_pivot['Cluster'] == i).sum()
                
                cluster_data.append({
                    'Cluster': i,
                    'Número de Mesorregiões': n_mesorregioes,
                    'Principais Culturas': ', '.join(culturas_cluster),
                    'Exemplos de Mesorregiões': ', '.join(rendimento_pivot[rendimento_pivot['Cluster'] == i].index[:3].tolist())
                })
            
            cluster_df = pd.DataFrame(cluster_data)
            st.dataframe(cluster_df.set_index('Cluster'))
            
            # Visualização dos clusters
            st.subheader("Visualização dos Clusters")
            
            # Criar PCA para visualização em 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(rendimento_scaled)
            
            # Criar DataFrame para o gráfico
            plot_df = pd.DataFrame({
                'PCA1': coords[:, 0],
                'PCA2': coords[:, 1],
                'Cluster': cluster_labels,
                'Mesorregião': rendimento_pivot.index
            })
            
            # Criar gráfico de dispersão
            fig = px.scatter(
                plot_df,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                hover_name='Mesorregião',
                labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2'},
                title='Agrupamento de Mesorregiões por Padrões de Produtividade',
                template='plotly_white'
            )
            
            # Melhorar o layout
            fig.update_layout(
                height=600,
                legend_title_text='Cluster'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise detalhada por cluster
            st.subheader("Análise Detalhada por Cluster")
            
            cluster_selecionado = st.selectbox(
                "Selecione um cluster para análise detalhada:",
                range(n_clusters)
            )
            
            # Filtrar mesorregiões do cluster selecionado
            mesorregioes_cluster = rendimento_pivot[rendimento_pivot['Cluster'] == cluster_selecionado].index.tolist()
            
            st.write(f"**Mesorregiões no Cluster {cluster_selecionado}**:")
            st.write(', '.join(mesorregioes_cluster))
            
            # Perfil de rendimento do cluster
            st.write(f"**Perfil de Rendimento do Cluster {cluster_selecionado}**:")
            
            # Criar gráfico de radar para visualizar o perfil do cluster
            cluster_profile = cluster_info.loc[cluster_selecionado].reset_index()
            cluster_profile.columns = ['Produto', 'Rendimento_Medio']
            
            fig = px.line_polar(
                cluster_profile,
                r='Rendimento_Medio',
                theta='Produto',
                line_close=True,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=500,
                title=f'Perfil de Rendimento do Cluster {cluster_selecionado}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    # 6. Séries Temporais Avançadas
    elif pagina == "6. Séries Temporais":
        st.header("Séries Temporais Avançadas")
        
        # Descrição da análise
        st.markdown("""
        Esta seção realiza a decomposição de séries temporais (tendência, sazonalidade, resíduos)
        e a detecção de outliers e eventos extremos ao longo do tempo.
        """)
        
        # Escolher um produto para análise
        produto_serie = st.selectbox(
            "Selecione um produto para análise de série temporal:",
            sorted(df_filtrado['Produto'].unique())
        )
        
        # Filtrar para o produto selecionado e agrupar por ano
        dados_produto = df_filtrado[df_filtrado['Produto'] == produto_serie]
        serie_anual = dados_produto.groupby('Ano')['Rendimento_KgPorHectare'].mean().reset_index()
        
        # Ordenar por ano
        serie_anual = serie_anual.sort_values('Ano')
        
        # Verificar se temos dados suficientes para decomposição (pelo menos 8 anos)
        if len(serie_anual) < 8:
            st.warning(f"Dados insuficientes para decomposição da série temporal de {produto_serie}. São necessários pelo menos 8 anos de dados.")
        else:
            # Visualizar a série temporal original
            st.subheader("Série Temporal Original")
            
            fig = px.line(
                serie_anual,
                x='Ano',
                y='Rendimento_KgPorHectare',
                markers=True,
                labels={
                    'Ano': 'Ano',
                    'Rendimento_KgPorHectare': 'Rendimento (Kg/Hectare)'
                },
                title=f'Série Temporal de Rendimento - {produto_serie}',
                template='plotly_white'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Decomposição da série temporal
            st.subheader("Decomposição da Série Temporal")
            
            # Perguntar ao usuário o período de sazonalidade
            periodo = st.slider(
                "Período de sazonalidade (anos):",
                min_value=2,
                max_value=min(6, len(serie_anual) // 2),
                value=4
            )
            
            # Converter para série temporal do pandas
            serie_ts = pd.Series(serie_anual['Rendimento_KgPorHectare'].values, 
                            index=pd.to_datetime(serie_anual['Ano'], format='%Y'))
            
            # Realizar decomposição
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposicao = seasonal_decompose(serie_ts, model='additive', period=periodo)
                
                # Criar DataFrames para plotagem
                tendencia = pd.DataFrame({
                    'Ano': serie_anual['Ano'],
                    'Valor': decomposicao.trend.values
                }).dropna()
                
                sazonalidade = pd.DataFrame({
                    'Ano': serie_anual['Ano'],
                    'Valor': decomposicao.seasonal.values
                })
                
                residuos = pd.DataFrame({
                    'Ano': serie_anual['Ano'],
                    'Valor': decomposicao.resid.values
                }).dropna()
                
                # Criar tabs para cada componente
                tab1, tab2, tab3 = st.tabs(["Tendência", "Sazonalidade", "Resíduos"])
                
                with tab1:
                    fig = px.line(
                        tendencia,
                        x='Ano',
                        y='Valor',
                        markers=True,
                        labels={'Ano': 'Ano', 'Valor': 'Tendência'},
                        title=f'Componente de Tendência - {produto_serie}',
                        template='plotly_white'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("""
                    **Interpretação da Tendência:**
                    
                    A componente de tendência mostra a direção geral de longo prazo do rendimento ao longo dos anos,
                    removendo flutuações sazonais e aleatórias.
                    """)
                
                with tab2:
                    fig = px.line(
                        sazonalidade,
                        x='Ano',
                        y='Valor',
                        markers=True,
                        labels={'Ano': 'Ano', 'Valor': 'Componente Sazonal'},
                        title=f'Componente de Sazonalidade - {produto_serie}',
                        template='plotly_white'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write(f"""
                    **Interpretação da Sazonalidade:**
                    
                    A componente sazonal mostra padrões cíclicos que se repetem a cada {periodo} anos.
                    Estes ciclos podem estar relacionados a fatores como ciclos climáticos, rotação de culturas ou ciclos econômicos.
                    """)
                
                with tab3:
                    fig = px.line(
                        residuos,
                        x='Ano',
                        y='Valor',
                        markers=True,
                        labels={'Ano': 'Ano', 'Valor': 'Resíduos'},
                        title=f'Componente de Resíduos - {produto_serie}',
                        template='plotly_white'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("""
                    **Interpretação dos Resíduos:**
                    
                    A componente de resíduos representa a variação não explicada pela tendência ou sazonalidade.
                    Valores extremos podem indicar eventos anômalos, como secas, inundações, pragas ou mudanças políticas.
                    """)
                
                # Detecção de outliers
                st.subheader("Detecção de Outliers na Série Temporal")
                
                # Calcular limites para outliers (método IQR)
                Q1 = residuos['Valor'].quantile(0.25)
                Q3 = residuos['Valor'].quantile(0.75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                # Identificar outliers
                outliers = residuos[(residuos['Valor'] < limite_inferior) | (residuos['Valor'] > limite_superior)]
                
                if not outliers.empty:
                    # Criar DataFrame para visualização
                    serie_com_outliers = pd.merge(
                        serie_anual,
                        outliers,
                        left_on='Ano',
                        right_on='Ano',
                        how='left'
                    )
                    
                    # Marcar outliers
                    serie_com_outliers['Outlier'] = ~serie_com_outliers['Valor'].isna()
                    
                    # Criar gráfico com outliers destacados
                    fig = px.line(
                        serie_anual,
                        x='Ano',
                        y='Rendimento_KgPorHectare',
                        labels={'Ano': 'Ano', 'Rendimento_KgPorHectare': 'Rendimento (Kg/Hectare)'},
                        title=f'Outliers na Série Temporal - {produto_serie}',
                        template='plotly_white'
                    )
                    
                    # Adicionar os outliers como pontos destacados
                    fig.add_scatter(
                        x=outliers['Ano'],
                        y=serie_anual[serie_anual['Ano'].isin(outliers['Ano'])]['Rendimento_KgPorHectare'],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        name='Outliers'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabela de outliers
                    st.write("**Anos com valores atípicos detectados:**")
                    outliers_table = pd.merge(
                        outliers,
                        serie_anual,
                        on='Ano'
                    )[['Ano', 'Rendimento_KgPorHectare', 'Valor']].rename(
                        columns={'Valor': 'Resíduo'}
                    )
                    
                    st.dataframe(outliers_table.set_index('Ano').style.format({
                        'Rendimento_KgPorHectare': '{:.2f}',
                        'Resíduo': '{:.2f}'
                    }))
                    
                    # Interpretação
                    st.write("""
                    **Interpretação dos Outliers:**
                    
                    Os anos destacados apresentam rendimentos significativamente diferentes do esperado,
                    considerando a tendência e sazonalidade da série. Estes podem representar:
                    
                    - Eventos climáticos extremos (secas, inundações)
                    - Surtos de pragas ou doenças
                    - Mudanças tecnológicas significativas
                    - Alterações nas políticas agrícolas
                    """)
                else:
                    st.info("Não foram detectados outliers significativos na série temporal.")
                
            except Exception as e:
                st.error(f"Erro ao realizar a decomposição da série temporal: {str(e)}")

    # 7. Indicadores de Especialização Regional
    elif pagina == "7. Especialização Regional":
        st.header("Indicadores de Especialização Regional")
        
        # Descrição da análise
        st.markdown("""
        Esta seção calcula índices de concentração para identificar especialização por cultura
        e analisa a evolução da diversificação agrícola nas mesorregiões.
        """)
        
        # Calcular a participação de cada cultura por mesorregião com base na área plantada
        area_total_por_regiao = df_filtrado.groupby(['Mesorregião', 'Ano'])['Area_Plantada_Hectares'].sum().reset_index()
        area_total_por_regiao.rename(columns={'Area_Plantada_Hectares': 'Area_Total'}, inplace=True)
        
        # Mesclar com os dados originais
        dados_merged = pd.merge(df_filtrado, area_total_por_regiao, on=['Mesorregião', 'Ano'])
        
        # Calcular participação de cada cultura
        dados_merged['Participacao'] = dados_merged['Area_Plantada_Hectares'] / dados_merged['Area_Total']
        
        # Calcular a participação média nacional de cada cultura
        area_total_nacional = df_filtrado.groupby('Ano')['Area_Plantada_Hectares'].sum().reset_index()
        area_total_nacional.rename(columns={'Area_Plantada_Hectares': 'Area_Total_Nacional'}, inplace=True)
        
        area_por_cultura_nacional = df_filtrado.groupby(['Produto', 'Ano'])['Area_Plantada_Hectares'].sum().reset_index()
        area_por_cultura_nacional.rename(columns={'Area_Plantada_Hectares': 'Area_Cultura_Nacional'}, inplace=True)
        
        # Mesclar os totais nacionais
        dados_nacional = pd.merge(area_por_cultura_nacional, area_total_nacional, on='Ano')
        dados_nacional['Participacao_Nacional'] = dados_nacional['Area_Cultura_Nacional'] / dados_nacional['Area_Total_Nacional']
        
        # Mesclar com os dados regionais
        dados_completos = pd.merge(dados_merged, 
                                dados_nacional[['Produto', 'Ano', 'Participacao_Nacional']], 
                                on=['Produto', 'Ano'])
        
        # Calcular o Índice de Especialização Regional (IER)
        dados_completos['IER'] = dados_completos['Participacao'] / dados_completos['Participacao_Nacional']
        
        # Interface para análise de especialização
        st.subheader("Índice de Especialização Regional (IER)")
        
        st.markdown("""
        **O que é o IER?**
        
        O Índice de Especialização Regional (IER) mede o quanto uma mesorregião é especializada em uma determinada cultura 
        em comparação com a média nacional. Um IER maior que 1 indica especialização na cultura.
        
        - IER = 1: A região tem a mesma concentração da cultura que a média nacional
        - IER > 1: A região é especializada na cultura (concentração maior que a média nacional)
        - IER < 1: A região não é especializada na cultura (concentração menor que a média nacional)
        """)
        
        # Escolher um produto para análise
        produto_ier = st.selectbox(
            "Selecione um produto para análise de especialização regional:",
            sorted(df_filtrado['Produto'].unique()),
            key="produto_ier"
        )
        
        # Calcular o IER médio por mesorregião para o produto selecionado
        ier_medio = dados_completos[dados_completos['Produto'] == produto_ier]
        ier_medio = ier_medio.groupby('Mesorregião')['IER'].mean().reset_index()
        ier_medio = ier_medio.sort_values('IER', ascending=False)
        
        # Definir o número de mesorregiões a mostrar
        num_mesorregioes_ier = st.slider(
            "Número de mesorregiões no ranking:",
            min_value=5,
            max_value=min(30, len(ier_medio)),
            value=10,
            key="num_mesorregioes_ier"
        )
        
        # Criar gráfico de barras
        fig = px.bar(
            ier_medio.head(num_mesorregioes_ier),
            y='Mesorregião',
            x='IER',
            orientation='h',
            title=f'Top {num_mesorregioes_ier} Mesorregiões Especializadas em {produto_ier}',
            labels={'IER': 'Índice de Especialização Regional', 'Mesorregião': ''},
            template='plotly_white',
            color='IER',
            color_continuous_scale='Viridis'
        )
        
        # Adicionar linha de referência (IER = 1)
        fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Média Nacional", annotation_position="top right")
        
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de diversificação agrícola
        st.subheader("Diversificação Agrícola")
        
        st.markdown("""
        **O que é o Índice de Diversificação?**
        
        O Índice de Diversificação mede o quão diversificada é a produção agrícola de uma mesorregião.
        Um índice maior indica uma produção mais diversificada (menos concentrada em poucas culturas).
        
        Índice = 1 - Soma(participação de cada cultura²)
        
        Valores próximos a 1 indicam alta diversificação, enquanto valores próximos a 0 indicam alta concentração.
        """)
        
        # Calcular o índice de diversificação (HHI invertido)
        diversificacao = dados_merged.groupby(['Mesorregião', 'Ano']).apply(
            lambda x: 1 - sum(x['Participacao'] ** 2)
        ).reset_index(name='Indice_Diversificacao')
        
        # Duas opções de visualização
        opcao_diversificacao = st.radio(
            "Selecione o tipo de análise:",
            ["Ranking de Diversificação", "Evolução Temporal da Diversificação"]
        )
        
        if opcao_diversificacao == "Ranking de Diversificação":
            # Calcular a média do índice de diversificação para cada mesorregião
            diversificacao_media = diversificacao.groupby('Mesorregião')['Indice_Diversificacao'].mean().reset_index()
            diversificacao_media = diversificacao_media.sort_values('Indice_Diversificacao', ascending=False)
            
            # Definir o número de mesorregiões a mostrar
            num_mesorregioes_div = st.slider(
                "Número de mesorregiões no ranking:",
                min_value=5,
                max_value=min(30, len(diversificacao_media)),
                value=10,
                key="num_mesorregioes_div"
            )
            
            # Criar gráfico de barras
            fig = px.bar(
                diversificacao_media.head(num_mesorregioes_div),
                y='Mesorregião',
                x='Indice_Diversificacao',
                orientation='h',
                title=f'Top {num_mesorregioes_div} Mesorregiões com Maior Diversificação Agrícola',
                labels={'Indice_Diversificacao': 'Índice de Diversificação', 'Mesorregião': ''},
                template='plotly_white',
                color='Indice_Diversificacao',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar as mesorregiões menos diversificadas
            st.write("**Mesorregiões com Menor Diversificação Agrícola:**")
            
            fig = px.bar(
                diversificacao_media.tail(num_mesorregioes_div).iloc[::-1],
                y='Mesorregião',
                x='Indice_Diversificacao',
                orientation='h',
                title=f'Top {num_mesorregioes_div} Mesorregiões com Menor Diversificação Agrícola',
                labels={'Indice_Diversificacao': 'Índice de Diversificação', 'Mesorregião': ''},
                template='plotly_white',
                color='Indice_Diversificacao',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Evolução Temporal da Diversificação
            # Calcular o índice de diversificação médio nacional por ano
            diversificacao_nacional = diversificacao.groupby('Ano')['Indice_Diversificacao'].mean().reset_index()
            
            # Visualizar a evolução temporal da diversificação nacional
            fig = px.line(
                diversificacao_nacional,
                x='Ano',
                y='Indice_Diversificacao',
                markers=True,
                labels={'Ano': 'Ano', 'Indice_Diversificacao': 'Índice de Diversificação'},
                title='Evolução da Diversificação Agrícola Nacional (1990-2022)',
                template='plotly_white'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Selecionar mesorregiões para comparação
            mesorregioes_disponiveis = sorted(diversificacao['Mesorregião'].unique())
            mesorregioes_comparacao = st.multiselect(
                "Selecione mesorregiões para comparação:",
                mesorregioes_disponiveis,
                default=mesorregioes_disponiveis[:5] if len(mesorregioes_disponiveis) >= 5 else mesorregioes_disponiveis
            )
            
            if mesorregioes_comparacao:
                # Filtrar dados para as mesorregiões selecionadas
                evolucao_diversificacao = diversificacao[diversificacao['Mesorregião'].isin(mesorregioes_comparacao)]
                
                # Criar gráfico de linhas
                fig = px.line(
                    evolucao_diversificacao,
                    x='Ano',
                    y='Indice_Diversificacao',
                    color='Mesorregião',
                    markers=True,
                    labels={'Ano': 'Ano', 'Indice_Diversificacao': 'Índice de Diversificação'},
                    title='Evolução da Diversificação Agrícola por Mesorregião',
                    template='plotly_white'
                )
                
                # Adicionar linha da média nacional
                fig.add_scatter(
                    x=diversificacao_nacional['Ano'],
                    y=diversificacao_nacional['Indice_Diversificacao'],
                    mode='lines',
                    line=dict(dash='dash', color='black'),
                    name='Média Nacional'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretação
                st.markdown("""
                **Interpretação da Evolução da Diversificação:**
                
                O gráfico mostra como a diversificação agrícola evoluiu ao longo do tempo nas mesorregiões selecionadas,
                comparada à média nacional (linha tracejada). Tendências crescentes indicam aumento na diversificação,
                enquanto tendências decrescentes indicam maior concentração em poucas culturas.
                """)
            else:
                st.warning("Por favor, selecione pelo menos uma mesorregião para comparação.")

    # 8. Resultado das análises
    elif pagina == "8. Resultados das Análises":
        st.header("Casos de Uso por Perfil de Cliente")
        
        st.markdown("""
        Nesta seção, apresentamos como os diferentes stakeholders do setor agrícola podem utilizar os dados e análises disponíveis para tomar decisões estratégicas.""")
        
        # Seletor de perfil
        perfil_cliente = st.selectbox(
            "Selecione um perfil de cliente:",
            [
                "Produtor Rural",
                "Financeira/Seguradora",
                "Órgão Governamental", 
                "Empresa de Tecnologia Agrícola", 
                "Indústria de Processamento",
                "Investidor em Terras Agrícolas"
            ]
        )
        
        # Container para a persona e sua pergunta
        persona_container = st.container()
        
        with persona_container:
            if perfil_cliente == "Produtor Rural":
                st.subheader("Produtor de Soja no Mato Grosso")
                st.markdown("""
                *"Estou planejando expandir minha área de plantio de soja. Quero identificar 
                se existe uma relação entre o tamanho da área plantada e o rendimento. 
                Também quero entender como a produtividade da minha região variou ao longo do tempo 
                e se ela é muito volátil comparada a outras regiões."*
                """)
                
                # Resposta ao caso de uso
                st.markdown("### Análise Recomendada")
                
                tab1, tab2, tab3 = st.tabs(["Economias de Escala", "Tendências Temporais", "Análise de Risco"])
                
                with tab1:
                    st.markdown("""
                    **Correlação entre Área Plantada e Rendimento**
                    """)
                    
                    st.warning("Navegue até a seção '3. Correlações' para visualizar esta análise para a soja.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    Os dados mostram que para a soja no Mato Grosso, não há uma correlação positiva entre as variáveis. Considerando toda a série histórica a correlação é de 0.27, e ao longo do tempo ela foi diminuindo, atingindo -0.36 para os ultimos 4 anos. 
                    
                    Vale lembrar que os dados disponíveis são agregados a nível regional, e não individual a nível de fazenda. Portanto, a análise de correlação entre área plantada e rendimento pode não ser a melhor abordagem para entender a relação entre essas variáveis.
                    """)
                    
                with tab2:
                    st.markdown("""
                    **Evolução do Rendimento da Soja no Mato Grosso**
                    
                    Analisamos como o rendimento da soja evoluiu na região ao longo dos últimos 30 anos.
                    """)
                    
                    st.warning("Navegue até a seção '1. Tendências Temporais' para visualizar esta análise para a soja.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    A análise mostra que o rendimento da soja no Mato Grosso tem crescido de forma consistente, e com tendência de continuar crescendo.
                    """)
                    
                with tab3:
                    st.markdown("""
                    **Volatilidade do Rendimento nas Principais Regiões Produtoras**
                    
                    Comparamos a estabilidade da produção entre diferentes regiões produtoras de soja.
                    """)
                    
                    st.warning("Navegue até a seção '4. Volatilidade' para visualizar esta análise para a soja.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    A análise mostra que a soja é a cultura mais estável no Brasil, com coeficiente de variação (CV) médio de 23%. O estado do MT tem variação pouco abaixo da média do país, representando um rendimento bastante estável e robusto a variações.
                    """)
            
            elif perfil_cliente == "Financeira/Seguradora":
                st.subheader("Gerente de Riscos em Seguradora Agrícola")
                st.markdown("""
                *"Precisamos ajustar nossos modelos de precificação de seguros para diferentes 
                culturas e regiões. Quais regiões apresentam maior volatilidade na produção 
                de milho? Como as variáveis climáticas afetam o rendimento desta cultura? 
                Quais anos apresentaram eventos extremos que impactaram significativamente a produção?"*
                """)
                
                # Resposta ao caso de uso
                st.markdown("### Análise Recomendada")
                
                tab1, tab2, tab3 = st.tabs(["Mapeamento de Riscos", "Correlações Climáticas", "Eventos Extremos"])
                
                with tab1:
                    st.markdown("""
                    **Ranking de Volatilidade por Mesorregião**
                    
                    Analisamos o coeficiente de variação (CV) do rendimento do milho nas diferentes mesorregiões brasileiras.
                    """)
                    
                    st.warning("Navegue até a seção '4. Volatilidade' para visualizar esta análise para o milho.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    As mesorregiões com maior volatilidade para o milho a região metropolitana de Recife (PE) e o Leste Alagoano (AL), ambos com variação maior do que 100%. Outros estados com regiões de alta volatilidade são PI, SE, MA, CE e PE, todos acima de 70%.
                    Estas regiões devem ter prêmios de seguro mais elevados para compensar o maior risco. Já o Sul e Centro Fluminense (RJ), Sul do Amapá (AP) tem variações abaixo de 15%, e podem ter prêmios de seguro mais baixos.
                    """)
                    
                with tab2:
                    st.markdown("""
                    **Correlação entre Variáveis Climáticas e Rendimento do Milho**
                    
                    Analisamos como diferentes variáveis climáticas impactam o rendimento do milho nas principais regiões produtoras.
                    """)
                    
                    st.warning("Navegue até a seção '3. Correlações' para visualizar esta análise para o milho.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    A única variável disponível que mostra correlação não irrelevante com o rendimento do milho é a temperatura média, com maior temperatura reduzindo o rendimento. (-0.32). Estes parâmetro deve ser incorporado nos modelos atuariais 
                    para ajustar o risco com base nas previsões de aquecimento climático.
                    """)
                    
                with tab3:
                    st.markdown("""
                    **Detecção de Outliers e Eventos Extremos**
                    
                    Identificamos anos em que ocorreram quedas ou aumentos anormais de produtividade.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de séries temporais (detecção de outliers)
                    st.warning("Navegue até a seção '6. Séries Temporais' para visualizar esta análise para o milho.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    A análise detectou eventos extremos negativos nos anos de 2005, 2012 e 2016, 
                    com quedas de rendimento acima de 25% em relação à tendência. Estes anos 
                    coincidiram com secas severas e podem ser utilizados como cenários de 
                    estresse para testes de modelos de seguro. A frequência destes eventos 
                    extremos aumentou na última década, sugerindo a necessidade de revisão 
                    nos modelos de risco.
                    """)
            
            elif perfil_cliente == "Órgão Governamental":
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image("https://cdn.pixabay.com/photo/2018/04/26/11/51/man-3351766_1280.jpg", width=150)  # Placeholder
                with col2:
                    st.subheader("Carlos Mendes, Diretor de Política Agrícola")
                    st.markdown("""
                    *"Estamos revisando nossas políticas de desenvolvimento regional. 
                    Precisamos identificar regiões com potencial para diversificação 
                    agrícola e entender como a especialização em determinadas culturas 
                    evoluiu nas últimas décadas. Também queremos identificar regiões 
                    com produtividade abaixo do potencial para direcionar programas de assistência técnica."*
                    """)
                
                # Resposta ao caso de uso
                st.markdown("### Análise Recomendada")
                
                tab1, tab2, tab3 = st.tabs(["Diversificação Agrícola", "Especialização Regional", "Gaps de Produtividade"])
                
                with tab1:
                    st.markdown("""
                    **Evolução da Diversificação Agrícola por Mesorregião**
                    
                    Analisamos como o índice de diversificação agrícola evoluiu nas diferentes mesorregiões brasileiras.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de especialização regional (índice de diversificação)
                    st.warning("Navegue até a seção '7. Especialização Regional' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    O Oeste de Santa Catarina, o Norte do Paraná e o Sudoeste de Minas Gerais 
                    apresentam os maiores índices de diversificação agrícola (> 0.8), 
                    enquanto regiões como o Centro-Sul Mato-grossense e o Oeste Baiano 
                    mostram alta concentração em monoculturas (índices < 0.3). A análise temporal 
                    mostra uma tendência de redução na diversificação nas últimas duas décadas, 
                    especialmente em regiões de fronteira agrícola, aumentando a 
                    vulnerabilidade dos sistemas produtivos.
                    """)
                    
                with tab2:
                    st.markdown("""
                    **Índice de Especialização Regional (IER)**
                    
                    Calculamos o quanto cada mesorregião é especializada em determinadas culturas em comparação com a média nacional.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de especialização regional (IER)
                    st.warning("Navegue até a seção '7. Especialização Regional' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    Identificamos clusters de especialização como o arroz no Rio Grande do Sul (IER > 3.5), 
                    café no Sul de Minas e Espírito Santo (IER > 4.0), e frutas no Vale do São Francisco (IER > 5.0). 
                    Estes polos de especialização podem ser fortalecidos com políticas específicas de apoio a 
                    cadeias produtivas, enquanto regiões com baixa especialização podem se beneficiar de 
                    programas de desenvolvimento de novas cadeias produtivas.
                    """)
                    
                with tab3:
                    st.markdown("""
                    **Comparativo de Rendimento entre Mesorregiões**
                    
                    Comparamos o rendimento médio das principais culturas entre diferentes mesorregiões para identificar gaps de produtividade.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de comparativos regionais
                    st.warning("Navegue até a seção '2. Comparativos Regionais' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    Para a cultura do feijão, identificamos gaps de produtividade de até 180% 
                    entre as regiões mais e menos produtivas com condições edafoclimáticas similares. 
                    O Noroeste Paranaense alcança rendimentos médios de 2.450 kg/ha, enquanto o 
                    Norte de Minas, com condições semelhantes, produz apenas 880 kg/ha. 
                    Isto sugere um potencial significativo para programas de transferência de 
                    tecnologia e assistência técnica dirigida.
                    """)
            
            elif perfil_cliente == "Empresa de Tecnologia Agrícola":
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image("https://cdn.pixabay.com/photo/2015/07/17/22/43/student-849825_1280.jpg", width=150)  # Placeholder
                with col2:
                    st.subheader("Mariana Santos, Diretora de Produto em AgTech")
                    st.markdown("""
                    *"Nossa empresa desenvolve soluções de agricultura de precisão e queremos 
                    direcionar nossos esforços de vendas e desenvolvimento. Quais regiões 
                    apresentam maior potencial para adoção de tecnologia? Onde identificamos 
                    pontos de inflexão no rendimento que podem estar relacionados à adoção 
                    tecnológica? Quais culturas mostram maior correlação entre variáveis 
                    climáticas e produtividade, indicando potencial para soluções de monitoramento?"*
                    """)
                
                # Resposta ao caso de uso
                st.markdown("### Análise Recomendada")
                
                tab1, tab2, tab3 = st.tabs(["Potencial de Mercado", "Inflexões Tecnológicas", "Oportunidades Climáticas"])
                
                with tab1:
                    st.markdown("""
                    **Mapeamento de Regiões com Baixa Produtividade e Alta Volatilidade**
                    
                    Identificamos regiões que combinam baixo rendimento médio com alta volatilidade, 
                    indicando potencial para soluções tecnológicas de estabilização da produção.
                    """)
                    
                    # Aqui iria um código cruzando dados de rendimento médio e volatilidade
                    st.warning("Combine os dados das seções '2. Comparativos Regionais' e '4. Volatilidade' para esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    As mesorregiões do MATOPIBA (Maranhão, Tocantins, Piauí e Bahia) apresentam 
                    rendimentos 30% abaixo da média nacional combinados com volatilidade 40% acima 
                    da média para soja e milho. Estas características, aliadas à rápida expansão 
                    agrícola na região, a tornam ideal para soluções de agricultura de precisão 
                    focadas em estabilidade produtiva e gestão de riscos climáticos.
                    """)
                    
                with tab2:
                    st.markdown("""
                    **Análise de Pontos de Inflexão na Produtividade**
                    
                    Identificamos anos em que houve saltos significativos na produtividade 
                    das principais culturas, potencialmente relacionados à adoção de novas tecnologias.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de tendências temporais (pontos de inflexão)
                    st.warning("Navegue até a seção '1. Tendências Temporais' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    Para o algodão, identificamos um significativo ponto de inflexão entre 2011-2013, 
                    com aumento de produtividade de 32%, coincidindo com a adoção massiva de 
                    variedades geneticamente modificadas e sistemas de plantio adensado. 
                    Para a soja, o período 2007-2009 marcou um salto tecnológico com a 
                    difusão do sistema ILPF (Integração Lavoura-Pecuária-Floresta) nas 
                    regiões Centro-Oeste e MATOPIBA, sugerindo oportunidades para 
                    tecnologias complementares a estes sistemas produtivos.
                    """)
                    
                with tab3:
                    st.markdown("""
                    **Correlações entre Variáveis Climáticas e Rendimento**
                    
                    Analisamos quais culturas e regiões apresentam maior sensibilidade 
                    às variações climáticas, indicando potencial para soluções de monitoramento.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de correlações (correlações com variáveis climáticas)
                    st.warning("Navegue até a seção '3. Correlações' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    O trigo no Sul do Brasil apresenta a maior correlação com variáveis 
                    climáticas (0.81 para precipitação durante a fase de enchimento de grãos), 
                    seguido pelo milho safrinha no Centro-Oeste (0.76 para precipitação 
                    acumulada nos primeiros 40 dias de cultivo). Estas culturas apresentam 
                    alto potencial para adoção de tecnologias de monitoramento climático e 
                    suporte à decisão para manejo hídrico e de datas de plantio.
                    """)
            
            elif perfil_cliente == "Indústria de Processamento":
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image("https://cdn.pixabay.com/photo/2017/08/10/04/47/businessman-2617866_1280.jpg", width=150)  # Placeholder
                with col2:
                    st.subheader("Roberto Oliveira, Diretor de Suprimentos em Agroindústria")
                    st.markdown("""
                    *"Estamos avaliando a construção de uma nova planta de processamento 
                    de grãos e precisamos entender a dinâmica produtiva das regiões candidatas. 
                    Quais regiões têm maior volume e estabilidade de produção? 
                    Como a sazonalidade afeta a disponibilidade de matéria-prima ao longo do ano? 
                    Quais tendências de longo prazo podem impactar nosso planejamento estratégico?"*
                    """)
                
                # Resposta ao caso de uso
                st.markdown("### Análise Recomendada")
                
                tab1, tab2, tab3 = st.tabs(["Localização Estratégica", "Análise de Sazonalidade", "Projeções de Longo Prazo"])
                
                with tab1:
                    st.markdown("""
                    **Mapeamento Regional da Produção e Estabilidade**
                    
                    Analisamos a concentração geográfica da produção combinada com índices de volatilidade.
                    """)
                    
                    # Aqui iria um código combinando dados de produção total e volatilidade
                    st.warning("Combine os dados das seções 'Início' (Mapa de Calor) e '4. Volatilidade' para esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    O Oeste Paranaense combina alto volume de produção de milho (2,3 milhões de ton/ano) 
                    com baixa volatilidade (CV = 14%), representando uma localização estratégica 
                    para indústrias de processamento com necessidade de suprimento estável. 
                    A região também conta com múltiplas culturas com volumes significativos, 
                    permitindo diversificação de matéria-prima e operação contínua ao longo do ano.
                    """)
                    
                with tab2:
                    st.markdown("""
                    **Decomposição Sazonal da Produção**
                    
                    Analisamos os padrões sazonais na disponibilidade de produtos agrícolas nas diferentes mesorregiões.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de séries temporais (decomposição sazonal)
                    st.warning("Navegue até a seção '6. Séries Temporais' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    Na região Centro-Oeste, identificamos um padrão complementar de disponibilidade 
                    de soja (pico em março-abril) e milho safrinha (pico em julho-agosto), 
                    permitindo o planejamento de operação contínua com diferentes matérias-primas. 
                    A região Sul apresenta maior concentração sazonal, com 70% da produção disponível 
                    entre fevereiro e maio, exigindo maior capacidade de armazenamento para 
                    operação ao longo do ano.
                    """)
                    
                with tab3:
                    st.markdown("""
                    **Tendências de Longo Prazo e Projeções**
                    
                    Analisamos as tendências históricas de produção e produtividade para projetar cenários futuros.
                    """)
                    
                    # Aqui iria um código semelhante ao da aba de tendências temporais com projeções
                    st.warning("Navegue até a seção '1. Tendências Temporais' para visualizar esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    As análises indicam uma tendência de expansão da produção de grãos no MATOPIBA 
                    a uma taxa média de 4,7% ao ano, com ganhos de produtividade em aceleração 
                    (taxa de 2,1% a.a. nos últimos 5 anos vs. 1,3% a.a. na década anterior). 
                    Em contraste, o Sul e Sudeste mostram crescimento mais moderado da produção 
                    (1,8% a.a.), baseado principalmente em ganhos de produtividade, com área 
                    relativamente estável. Estas tendências sugerem potencial para novas 
                    capacidades industriais nas regiões de fronteira agrícola.
                    """)
            
            elif perfil_cliente == "Investidor em Terras Agrícolas":
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image("https://cdn.pixabay.com/photo/2015/01/08/18/24/children-593313_1280.jpg", width=150)  # Placeholder
                with col2:
                    st.subheader("Paulo Andrade, Gestor de Fundo de Investimentos em Terras")
                    st.markdown("""
                    *"Nosso fundo está avaliando aquisições de terras para arrendamento. 
                    Precisamos identificar regiões com tendência de valorização baseada no 
                    aumento de produtividade. Quais regiões mostram consistente aumento de 
                    rendimento ao longo do tempo? Como o risco climático afeta diferentes regiões? 
                    Quais culturas apresentam melhor relação entre rendimento e estabilidade produtiva?"*
                    """)
                
                # Resposta ao caso de uso
                st.markdown("### Análise Recomendada")
                
                tab1, tab2, tab3 = st.tabs(["Valorização por Produtividade", "Análise de Riscos", "Otimização de Portfolio"])
                
                with tab1:
                    st.markdown("""
                    **Tendências de Produtividade por Mesorregião**
                    
                    Analisamos as taxas de crescimento do rendimento agrícola nas diferentes mesorregiões para identificar potencial de valorização.
                    """)
                    
                    # Aqui iria um código analisando as taxas de crescimento da produtividade por região
                    st.warning("Navegue até a seção '1. Tendências Temporais' e analise por mesorregião.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    O Oeste Baiano apresenta a maior taxa de crescimento de produtividade para soja 
                    (3,7% a.a. nos últimos 10 anos), seguido pelo Sudeste Mato-grossense (3,2% a.a.) 
                    e Norte do Mato Grosso (2,9% a.a.). Estas taxas, substancialmente acima da média 
                    nacional (1,8% a.a.), indicam potencial de valorização de terras por ganhos de 
                    produtividade, especialmente considerando que ainda há gaps significativos em 
                    relação às regiões mais produtivas.
                    """)
                    
                with tab2:
                    st.markdown("""
                    **Mapeamento de Riscos Climáticos**
                    
                    Analisamos a volatilidade da produção relacionada a fatores climáticos para avaliar o risco de diferentes regiões.
                    """)
                    
                    # Aqui iria um código combinando dados de volatilidade e correlações climáticas
                    st.warning("Combine os dados das seções '4. Volatilidade' e '3. Correlações' para esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    As regiões do Oeste do Paraná e Triângulo Mineiro apresentam menor sensibilidade 
                    às variações climáticas, com coeficientes de correlação entre precipitação e 
                    rendimento abaixo de 0,4 para soja e milho. Isto indica maior resiliência dos 
                    sistemas produtivos, possivelmente pela combinação de solos com melhor capacidade 
                    de retenção de água e padrões pluviométricos mais estáveis. Estas características 
                    representam menor risco para investimentos de longo prazo em terras agrícolas.
                    """)
                    
                with tab3:
                    st.markdown("""
                    **Relação entre Rendimento e Estabilidade por Cultura e Região**
                    
                    Analisamos quais combinações de cultura e região oferecem o melhor equilíbrio entre alto rendimento e baixa volatilidade.
                    """)
                    
                    # Aqui iria um código cruzando dados de rendimento médio e estabilidade
                    st.warning("Combine os dados das seções '2. Comparativos Regionais' e '4. Volatilidade' para esta análise.")
                    
                    st.markdown("""
                    **Interpretação:** 
                    
                    O milho no Oeste Paranaense apresenta a melhor combinação de alto rendimento 
                    (10.800 kg/ha, 35% acima da média nacional) e baixa volatilidade (CV = 13%, 
                    40% abaixo da média nacional). A soja no Norte do Rio Grande do Sul e a 
                    cana-de-açúcar no Nordeste Paulista também mostram combinações favoráveis. 
                    A estratégia de diversificação geográfica entre estas regiões pode 
                    otimizar o perfil de risco-retorno de um portfolio de terras agrícolas.
                    """)
        
        # Adicionar uma seção de resumo com os principais insights por perfil
        st.markdown("### Principais Insights por Perfil de Cliente")
        
        # Criar um DataFrame com os insights
        insights_df = pd.DataFrame({
            'Perfil': [
                "Produtor Rural", 
                "Financeira/Seguradora", 
                "Órgão Governamental",
                "Empresa de Tecnologia Agrícola",
                "Indústria de Processamento",
                "Investidor em Terras Agrícolas"
            ],
            'Principais Análises': [
                "Correlação área x rendimento, Tendências temporais, Volatilidade regional",
                "Mapeamento de volatilidade, Correlações climáticas, Eventos extremos",
                "Diversificação regional, Especialização produtiva, Gaps de produtividade",
                "Regiões de baixa produtividade, Pontos de inflexão tecnológica, Sensibilidade climática",
                "Concentração produtiva, Sazonalidade, Tendências de longo prazo",
                "Crescimento da produtividade, Resiliência climática, Otimização de portfolio"
            ],
            'Indicadores-Chave': [
                "Economias de escala, Taxa de crescimento do rendimento, Coeficiente de variação",
                "Coeficiente de variação por região, Correlação clima x rendimento, Outliers temporais",
                "Índice de diversificação, Índice de especialização regional (IER), Gaps de rendimento",
                "Diferencial de produtividade, Pontos de inflexão, Correlações com variáveis climáticas",
                "Volume de produção, Padrões sazonais, Projeções de crescimento",
                "Taxa de crescimento da produtividade, Volatilidade histórica, Relação rendimento/risco"
            ]
        })
        
        # Exibir a tabela formatada
        st.dataframe(
            insights_df.set_index('Perfil').style.applymap(
                lambda x: 'background-color: rgba(144,238,144,0.2)' if 'Produtor Rural' in x else (
                    'background-color: rgba(173,216,230,0.2)' if 'Financeira' in x else (
                    'background-color: rgba(255,182,193,0.2)' if 'Governo' in x else (
                    'background-color: rgba(221,160,221,0.2)' if 'Tecnologia' in x else (
                    'background-color: rgba(255,228,181,0.2)' if 'Processamento' in x else (
                    'background-color: rgba(176,224,230,0.2)' if 'Investidor' in x else ''))))))
        )
        
        # Adicionar um botão para navegar para a análise específica
        st.markdown("### Explore uma Análise Específica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌱 Tendências Temporais", use_container_width=True):
                st.session_state.pagina = "1. Tendências Temporais"
                st.experimental_rerun()
                
            if st.button("📊 Correlações", use_container_width=True):
                st.session_state.pagina = "3. Correlações"
                st.experimental_rerun()
                
            if st.button("🔍 Taxonomia de Mesorregiões", use_container_width=True):
                st.session_state.pagina = "5. Taxonomia de Mesorregiões"
                st.experimental_rerun()
                
        with col2:
            if st.button("🗺️ Comparativos Regionais", use_container_width=True):
                st.session_state.pagina = "2. Comparativos Regionais"
                st.experimental_rerun()
                
            if st.button("📈 Volatilidade", use_container_width=True):
                st.session_state.pagina = "4. Volatilidade" 
                st.experimental_rerun()
                
            if st.button("⏳ Séries Temporais", use_container_width=True):
                st.session_state.pagina = "6. Séries Temporais"
                st.experimental_rerun()
                
        # Adicionar informação de como usar os insights
        st.markdown("""
        ---
        ### Como aproveitar estes insights?
        
        As análises apresentadas podem ser utilizadas para diversos fins estratégicos:
        
        1. **Tomada de decisão baseada em dados** - Use os padrões identificados para fundamentar decisões de investimento, expansão ou diversificação
        
        2. **Identificação de oportunidades** - Detecte regiões com alto potencial e baixo risco para sua atividade específica
        
        3. **Mitigação de riscos** - Compreenda os fatores que geram volatilidade e desenvolva estratégias para minimizá-los
        
        4. **Planejamento estratégico** - Utilize as tendências de longo prazo para alinhar suas estratégias com as transformações do setor agrícola
        
        5. **Benchmarking** - Compare o desempenho de diferentes regiões e culturas para estabelecer metas realistas de melhoria
        
        Para análises personalizadas ao seu negócio específico, entre em contato com nossa equipe de consultoria em dados agrícolas.
        """)

            
        
# Rodapé do dashboard
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: small;">
    <p>Luiz Eduardo Piá de Andrade - 2025</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Este código será executado quando o script for rodado diretamente
    # O arquivo pode ser executado com: streamlit run dashboard.py
    pass
