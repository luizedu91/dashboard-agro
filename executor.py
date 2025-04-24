# executor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Importar o módulo de análises
from analises_agricolas import executar_analises

def main():
    """
    Função principal para carregar os dados e executar as análises
    """
    print("Iniciando o programa de análises agrícolas...\n")
    
    df_consolidado = pd.read_parquet('dados_meteo.parquet')
    
    print(f"Dados carregados com sucesso! Dimensões: {df_consolidado.shape}")
    print(f"Colunas disponíveis: {', '.join(df_consolidado.columns)}")
    
    # Verificar se as colunas necessárias estão presentes
    colunas_necessarias = [
        'Mesorregião', 'Ano', 'Produto', 'Area_Plantada_Hectares', 
        'Producao_Toneladas', 'Rendimento_KgPorHectare', 'Valor_Produzido_Mil_Reais'
    ]
    
    colunas_faltantes = [col for col in colunas_necessarias if col not in df_consolidado.columns]
    
    if colunas_faltantes:
        print(f"Erro: As seguintes colunas necessárias não foram encontradas: {', '.join(colunas_faltantes)}")
        return
    
    # Verificar se há variáveis climáticas
    var_climaticas = [col for col in df_consolidado.columns if col in [
        'precipitacao_total_anual', 'radiacao_global_media', 
        'temperatura_bulbo_media', 'vento_velocidade_media'
    ]]
    
    if var_climaticas:
        print(f"Variáveis climáticas encontradas: {', '.join(var_climaticas)}")
    else:
        print("Atenção: Nenhuma variável climática foi encontrada nos dados.")
    
    # Mostrar informações básicas sobre os dados
    print("\nInformações básicas sobre os dados:")
    print(f"Período: {df_consolidado['Ano'].min()} a {df_consolidado['Ano'].max()}")
    print(f"Número de mesorregiões: {df_consolidado['Mesorregião'].nunique()}")
    print(f"Produtos: {', '.join(sorted(df_consolidado['Produto'].unique()))}")
    
    # Verificar missings
    missings = df_consolidado.isnull().sum()
    if missings.sum() > 0:
        print("\nAtenção: Foram encontrados valores missing nos dados:")
        print(missings[missings > 0])
        
        # Perguntar se deseja continuar
        continuar = input("\nDeseja continuar mesmo com valores missing? (s/n): ")
        if continuar.lower() != 's':
            print("Programa encerrado pelo usuário.")
            return
    
    # Executar as análises
    print("\nIniciando execução das análises...")
    resultados = executar_analises(df_consolidado)
    
    print("\nTodas as análises foram concluídas!")
    print("Os resultados foram salvos na pasta 'figuras' e respectivas subpastas.")
    
    # Perguntar se deseja iniciar o dashboard
    iniciar_dashboard = input("\nDeseja iniciar o dashboard interativo? (s/n): ")
    if iniciar_dashboard.lower() == 's':
        print("\nIniciando o dashboard interativo...")
        print("Pressione Ctrl+C para encerrar o dashboard quando desejar.")
        os.system("streamlit run dashboard.py")

if __name__ == "__main__":
    main()