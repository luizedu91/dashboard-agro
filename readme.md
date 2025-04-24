# Dashboard de Análises Agrícolas

Este projeto apresenta um dashboard interativo para análise de dados agrícolas brasileiros, focando na produtividade, produção e área plantada de diferentes culturas (milho, arroz, soja, feijão e trigo) ao longo do período de 1990 a 2022, com dados meteorológicos integrados.

## Sobre o Dashboard

O dashboard foi desenvolvido usando Streamlit para criar uma interface interativa e intuitiva que permite explorar relações entre fatores climáticos, eventos técnicos e a produtividade agrícola nas diferentes mesorregiões do Brasil.

## Funcionalidades

O dashboard está organizado em seções temáticas acessíveis pelo menu lateral:

### 1. Tendências Temporais
- Visualização da evolução do rendimento médio por cultura 
- Identificação de pontos de inflexão na produtividade (anos com mudanças significativas)
- Análise customizável com limiar de variação ajustável (padrão: 15%)

### 2. Comparativos Regionais
- Ranking das mesorregiões mais produtivas para cada cultura
- Mapas de calor mostrando a distribuição espacial da produtividade
- Visualizações por produto ou por mesorregião

### 3. Correlações
- Relação entre área plantada e rendimento (análise de economias de escala)
- Correlação entre valor da produção e rendimento
- Correlações com variáveis climáticas (precipitação, temperatura, radiação, vento)

### 4. Volatilidade
- Análise do coeficiente de variação do rendimento por cultura e região
- Identificação das regiões e culturas mais estáveis/instáveis

### 5. Taxonomia de Mesorregiões
- Agrupamento de regiões com padrões similares de produtividade
- Classificação por perfil de culturas predominantes

### 6. Séries Temporais
- Decomposição das séries (tendência, sazonalidade, resíduos)
- Detecção de outliers e eventos extremos

### 7. Especialização Regional
- Índices de concentração para identificar especialização por cultura
- Evolução da diversificação agrícola nas mesorregiões

## Fontes de Dados

- **Dados Agrícolas**: IBGE - Pesquisa Agrícola Municipal (PAM), com dados a nível de mesorregião
- **Dados Climáticos**: INMET - Instituto Nacional de Meteorologia, via BDMEP (Banco de Dados Meteorológicos para Ensino e Pesquisa)
- **Dados Geográficos**: Biblioteca GeoBR para shapes das mesorregiões brasileiras

## Instalação e Execução

1. Clone este repositório
2. Instale as dependências:
```
pip install -r requirements.txt
```
3. Execute o dashboard:
```
streamlit run dashboard.py
```

## Estrutura do Projeto

- `dashboard.py`: Aplicação principal Streamlit
- `Safra.ipynb`: Notebook com aquisição dos dados a partir de APIs, transformação e limpeza deles, assim exploração e análises preliminares
- `dados_agricolas_mesorregiao.parquet`: Dados consolidados de produção agrícola
- `dados_meteo.parquet`: Dados agriolas consolidados com dados meteorológicos

## Insights Destacados

- Análise do impacto de eventos climáticos (El Niño/La Niña) na produtividade
- Identificação de saltos tecnológicos na produtividade de culturas específicas
- Visualização de padrões regionais de especialização agrícola
- Correlações entre variáveis climáticas e desempenho agrícola

## Limitações e Trabalhos Futuros

- Expandir análise para outras culturas importantes
- Implementar modelos preditivos de produtividade com base em dados climáticos
- Incorporar análises de impacto econômico e sustentabilidade

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias.

## Licença

Este projeto está licenciado sob a licença MIT - consulte o arquivo LICENSE para mais detalhes.