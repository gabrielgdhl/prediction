import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from dados import carregar_resultados
from features import GeradorEstatisticasAvancadas


def treinar_modelo(X, y):
    print("Treinando modelo de Machine Learning...")

    modelo = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )

    modelo.fit(X, y)

    return modelo


def obter_probabilidade_nao_sair(modelo, features):
    """
    Retorna um ranking das 25 dezenas
    pela probabilidade de NÃO serem sorteadas.
    """

    features = features.reshape(1, -1)

    probabilidades = modelo.predict_proba(features)

    ranking = []

    for indice_dezena in range(25):

        classes = modelo.estimators_[indice_dezena].classes_

        probs = probabilidades[indice_dezena][0]

        if 0 in classes:
            indice_classe_zero = np.where(classes == 0)[0][0]
            prob_nao_sair = probs[indice_classe_zero]
        else:
            prob_nao_sair = 0.0

        ranking.append({
            "dezena": indice_dezena + 1,
            "prob_nao_sair": float(prob_nao_sair)
        })

    ranking.sort(
        key=lambda item: item["prob_nao_sair"],
        reverse=True
    )

    return ranking


def selecionar_exclusoes(ranking, quantidade=7):
    exclusoes = [
        item["dezena"]
        for item in ranking[:quantidade]
    ]

    return sorted(exclusoes)


def mostrar_ranking(ranking):
    print()
    print("=" * 50)
    print("RANKING DE PROBABILIDADE DE NÃO SAIR")
    print("=" * 50)

    for posicao, item in enumerate(ranking, start=1):

        print(
            f"{posicao:02d}º | "
            f"Dezena {item['dezena']:02d} | "
            f"P(não sair): "
            f"{item['prob_nao_sair']:.4f}"
        )


def main():
    print("=" * 60)
    print("LOTOFÁCIL - MODELO PREDITIVO DE EXCLUSÃO")
    print("=" * 60)

    # ========================================================
    # 1. CARREGAR DADOS
    # ========================================================

    print()
    print("1. Carregando dados históricos da Lotofácil...")

    try:

        df, df_bolas = carregar_resultados()

        print(
            f"-> Total de concursos carregados: "
            f"{len(df_bolas)}"
        )

    except Exception as e:

        print(
            f"Erro ao carregar os dados: {e}"
        )

        return

    # ========================================================
    # 2. GERAR FEATURES
    # ========================================================

    print()
    print(
        "2. Inicializando Engenharia "
        "de Features Estatísticas..."
    )

    gerador = GeradorEstatisticasAvancadas(
        df_bolas
    )

    print(
        f"-> Total de concursos: "
        f"{gerador.total_sorteios}"
    )

    # ========================================================
    # 3. CRIAR DATASET
    # ========================================================

    print()
    print(
        "3. Construindo dataset de treinamento..."
    )

    X, y, indices = gerador.construir_dataset_ml(
        janela_minima=200
    )

    print(
        f"-> X.shape: {X.shape}"
    )

    print(
        f"-> y.shape: {y.shape}"
    )

    # ========================================================
    # 4. TREINAR MODELO
    # ========================================================

    print()
    print(
        "4. Treinando modelo..."
    )

    modelo = treinar_modelo(
        X,
        y
    )

    print(
        "-> Modelo treinado."
    )

    # ========================================================
    # 5. FEATURES DO PRÓXIMO CONCURSO
    # ========================================================

    print()
    print(
        "5. Gerando features para "
        "o próximo concurso..."
    )

    features_proximo = (
        gerador.features_proximo_concurso()
    )

    # ========================================================
    # 6. RANKING
    # ========================================================

    ranking = obter_probabilidade_nao_sair(
        modelo,
        features_proximo
    )

    mostrar_ranking(
        ranking
    )

    # ========================================================
    # 7. SELECIONAR 7 EXCLUSÕES
    # ========================================================

    exclusoes = selecionar_exclusoes(
        ranking,
        quantidade=7
    )

    todas_dezenas = set(
        range(1, 26)
    )

    candidatas = sorted(
        todas_dezenas
        - set(exclusoes)
    )

    print()
    print("=" * 50)
    print("7 DEZENAS PARA EXCLUSÃO")
    print("=" * 50)

    print(
        " ".join(
            f"{d:02d}"
            for d in exclusoes
        )
    )

    print()
    print("=" * 50)
    print("18 DEZENAS CANDIDATAS")
    print("=" * 50)

    print(
        " ".join(
            f"{d:02d}"
            for d in candidatas
        )
    )


if __name__ == "__main__":
    main()