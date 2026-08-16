import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from features import GeradorEstatisticasAvancadas

def carregar_dados(caminho_csv):
    """
    Espera um CSV onde cada linha é um concurso e as colunas contêm as 15 dezenas sorteadas.
    Ajuste os nomes das colunas conforme o seu CSV real.
    """
    df = pd.read_csv(caminho_csv)
    # Exemplo assumindo colunas de B1 a B15 ou similar
    colunas_bolas = [col for col in df.columns if 'bola' in col.lower() or 'dezena' in col.lower() or col.startswith('B')]
    if not colunas_bolas:
        # Se as primeiras 15 colunas forem os números
        colunas_bolas = df.columns[1:16]
    
    return df[colunas_bolas]

def gerar_jogos_otimizados(exclusoes_previstas, qtd_jogos=3, tamanho_jogo=15):
    """
    Gera combinações de jogos de 15 dezenas excluindo as piores apontadas pelo ML.
    """
    todas_dezenas = set(range(1, 26))
    dezenas_disponiveis = list(todas_dezenas - set(exclusoes_previstas))
    
    jogos = []
    np.random.seed(42)
    
    for _ in range(qtd_jogos):
        # Seleciona aleatoriamente dentro do pool limpo e ordena
        jogo = sorted(np.random.choice(dezenas_disponiveis, size=tamanho_jogo, replace=False))
        jogos.append(jogo)
        
    return jogos

if __name__ == "__main__":
    print("1. Carregando dados históricos da Lotofácil...")
    # Substitua pelo nome do seu arquivo CSV real
    caminho_csv = "dados_lotofacil.csv" 
    
    try:
        df_bolas = carregar_dados(caminho_csv)
        print(f"-> Total de concursos carregados: {len(df_bolas)}")
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        print("Certifique-se de que o arquivo 'dados_lotofacil.csv' está na mesma pasta.")
        exit()

    print("\n2. Inicializando Engenharia de Features Estatísticas...")
    gerador = GeradorEstatisticasAvancadas(df_bolas)

    print("\n3. Construindo dataset de treinamento (X, y)...")
    # Usa os concursos a partir do índice 200 para garantir histórico suficiente nas janelas
    X, y = gerador.construir_dataset_ml(janela_minima=200)
    print(f"-> Amostras de treino criadas: {X.shape[0]}")

    print("\n4. Treinando o modelo de Machine Learning (Random Forest Multi-Output)...")
    # Treina com os dados históricos passados
    modelo = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, n_jobs=-1)
    )
    modelo.fit(X, y)
    print("-> Modelo treinado com sucesso!")

    print("\n5. Calculando previsões e exclusões para o próximo concurso...")
    # Pega o índice do último concurso disponível na base
    ultimo_indice = len(df_bolas) - 1
    features_atual = gerador.calcular_features_no_indice(ultimo_indice).reshape(1, -1)
    
    probabilidades = modelo.predict_proba(features_atual)
    
    ranking_nao_sair = []
    for dezena in range(25):
        # Probabilidade da classe 0 (probabilidade da dezena NÃO sair no próximo)
        prob_nao_sair = probabilidades[dezena][0][0] if len(probabilidades[dezena][0]) > 1 else 0.5
        ranking_nao_sair.append((dezena + 1, prob_nao_sair))
        
    # Ordena do maior risco de NÃO sair para o menor
    ranking_nao_sair.sort(key=lambda x: x[1], reverse=True)

    # Vamos sugerir excluir as 7 piores dezenas preditas pelo modelo
    n_exclusoes = 7
    exclusoes = [d for d, p in ranking_nao_sair[:n_exclusoes]]
    
    print("\n" + "="*50)
    print(f" RESULTADO DA ANÁLISE PARA O PRÓXIMO CONCURSO ")
    print("="*50)
    print(f"🚫 Piores dezenas sugeridas para EXCLUSÃO ({n_exclusoes} dezenas):")
    for dezena, prob in ranking_nao_sair[:n_exclusoes]:
        print(f"   - Dezena {dezena:02d} (Probabilidade de falha: {prob*100:.1f}%)")

    print("\n" + "-"*50)
    print("🎲 Sugestões de Jogos Otimizados (Filtrando as exclusões):")
    meus_jogos = gerar_jogos_otimizados(exclusoes, qtd_jogos=3, tamanho_jogo=15)
    for i, jogo in enumerate(meus_jogos, 1):
        jogo_str = " ".join([f"{num:02d}" for num in jogo])
        print(f"   Jogo {i}: [ {jogo_str} ]")
    print("="*50)