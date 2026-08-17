import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from dados import carregar_resultados
from features import GeradorEstatisticasAvancadas

from gerador_jogos import (
    GeradorJogos,
    imprimir_jogos,
    calcular_custo
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

QUANTIDADE_EXCLUSOES = 4

QUANTIDADE_JOGOS = 50

JANELA_MINIMA_TREINO = 200

PRECO_APOSTA = 3.50


# ============================================================
# TREINAMENTO
# ============================================================

def treinar_modelo(X, y):
    """
    Treina 25 classificadores independentes.

    Para cada dezena:

        classe 1 = saiu
        classe 0 = não saiu
    """

    print()
    print("4. Treinando modelo de Machine Learning...")

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

    modelo.fit(
        X,
        y
    )

    print("-> Modelo treinado.")

    return modelo


# ============================================================
# PROBABILIDADES
# ============================================================

def obter_ranking_nao_sair(
    modelo,
    features
):
    """
    Calcula P(não sair) para cada uma das 25 dezenas.

    Retorna:

        [
            {
                "dezena": 7,
                "prob_nao_sair": 0.72
            },
            ...
        ]
    """

    features = (
        np.asarray(
            features
        )
        .reshape(1, -1)
    )

    probabilidades = (
        modelo.predict_proba(
            features
        )
    )

    ranking = []

    for indice_dezena in range(25):

        estimador = (
            modelo.estimators_[
                indice_dezena
            ]
        )

        classes = (
            estimador.classes_
        )

        probs = (
            probabilidades[
                indice_dezena
            ][0]
        )

        # ================================================
        # Queremos explicitamente:
        #
        # P(classe = 0)
        #
        # onde:
        #
        # 0 = NÃO saiu
        # 1 = saiu
        # ================================================

        if 0 in classes:

            indice_classe_zero = (
                np.where(
                    classes == 0
                )[0][0]
            )

            prob_nao_sair = float(
                probs[
                    indice_classe_zero
                ]
            )

        else:

            prob_nao_sair = 0.0

        ranking.append({
            "dezena":
                indice_dezena + 1,

            "prob_nao_sair":
                prob_nao_sair
        })

    ranking.sort(
        key=lambda item:
            item["prob_nao_sair"],
        reverse=True
    )

    return ranking


# ============================================================
# EXCLUSÕES
# ============================================================

def selecionar_exclusoes(
    ranking,
    quantidade
):
    """
    Seleciona as N dezenas com maior
    probabilidade estimada de NÃO sair.
    """

    exclusoes = [
        item["dezena"]
        for item
        in ranking[:quantidade]
    ]

    return sorted(
        exclusoes
    )


# ============================================================
# OUTPUT
# ============================================================

def mostrar_ranking(
    ranking
):
    print()
    print("=" * 65)
    print(
        "RANKING DE PROBABILIDADE DE NÃO SAIR"
    )
    print("=" * 65)

    for posicao, item in enumerate(
        ranking,
        start=1
    ):

        print(
            f"{posicao:02d}º | "
            f"Dezena {item['dezena']:02d} | "
            f"P(não sair): "
            f"{item['prob_nao_sair']:.4f}"
        )


def mostrar_dezenas(
    titulo,
    dezenas
):
    print()
    print("=" * 65)
    print(titulo)
    print("=" * 65)

    print(
        " ".join(
            f"{dezena:02d}"
            for dezena
            in dezenas
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print(
        "LOTOFÁCIL - MODELO DE EXCLUSÃO "
        "+ FECHAMENTO"
    )
    print("=" * 65)

    # ========================================================
    # 1. CARREGAR HISTÓRICO
    # ========================================================

    print()
    print(
        "1. Carregando dados históricos..."
    )

    try:

        df, df_bolas = (
            carregar_resultados()
        )

    except Exception as erro:

        print()
        print(
            "ERRO AO CARREGAR OS DADOS:"
        )

        print(
            erro
        )

        return

    print(
        f"-> Concursos carregados: "
        f"{len(df_bolas)}"
    )

    # ========================================================
    # 2. FEATURES
    # ========================================================

    print()
    print(
        "2. Inicializando estatísticas..."
    )

    gerador_stats = (
        GeradorEstatisticasAvancadas(
            df_bolas
        )
    )

    print(
        f"-> Features por dezena: "
        f"{gerador_stats.quantidade_features_por_dezena()}"
    )

    print(
        f"-> Features totais: "
        f"{gerador_stats.quantidade_features_total()}"
    )

    # ========================================================
    # 3. DATASET ML
    # ========================================================

    print()
    print(
        "3. Construindo dataset..."
    )

    X, y, indices = (
        gerador_stats
        .construir_dataset_ml(
            janela_minima=
                JANELA_MINIMA_TREINO
        )
    )

    print(
        f"-> X.shape: {X.shape}"
    )

    print(
        f"-> y.shape: {y.shape}"
    )

    print(
        f"-> Exemplos de treino: "
        f"{len(X)}"
    )

    # ========================================================
    # 4. TREINAMENTO
    # ========================================================

    modelo = treinar_modelo(
        X,
        y
    )

    # ========================================================
    # 5. PRÓXIMO CONCURSO
    # ========================================================

    print()
    print(
        "5. Calculando estatísticas "
        "para o próximo concurso..."
    )

    features_proximo = (
        gerador_stats
        .features_proximo_concurso()
    )

    # ========================================================
    # 6. RANKING
    # ========================================================

    ranking = (
        obter_ranking_nao_sair(
            modelo,
            features_proximo
        )
    )

    mostrar_ranking(
        ranking
    )

    # ========================================================
    # 7. EXCLUSÕES
    # ========================================================

    exclusoes = (
        selecionar_exclusoes(
            ranking,
            QUANTIDADE_EXCLUSOES
        )
    )

    todas_dezenas = set(
        range(1, 26)
    )

    candidatas = sorted(
        todas_dezenas
        - set(exclusoes)
    )

    mostrar_dezenas(
        f"{QUANTIDADE_EXCLUSOES} "
        f"DEZENAS PARA EXCLUSÃO",
        exclusoes
    )

    mostrar_dezenas(
        f"{len(candidatas)} "
        f"DEZENAS CANDIDATAS",
        candidatas
    )

    # ========================================================
    # 8. GERADOR DE JOGOS
    # ========================================================

    print()
    print("=" * 65)
    print(
        "GERANDO FECHAMENTO"
    )
    print("=" * 65)

    # Por enquanto:
    #
    # SEM filtros.
    #
    # Primeiro queremos avaliar o fechamento
    # somente por diversidade.
    gerador_jogos = (
        GeradorJogos(
            filtros=[]
        )
    )

    try:

        resultado_fechamento = (
            gerador_jogos.gerar(
                dezenas_candidatas=
                    candidatas,

                quantidade_jogos=
                    QUANTIDADE_JOGOS,

                stats=None
            )
        )

    except Exception as erro:

        print()
        print(
            "ERRO AO GERAR FECHAMENTO:"
        )

        print(
            erro
        )

        return

    # ========================================================
    # 9. ESTATÍSTICAS DO FECHAMENTO
    # ========================================================

    total_combinacoes = (
        resultado_fechamento[
            "total_combinacoes"
        ]
    )

    total_apos_filtros = (
        resultado_fechamento[
            "total_apos_filtros"
        ]
    )

    jogos = (
        resultado_fechamento[
            "jogos"
        ]
    )

    print()
    print(
        f"Combinações possíveis: "
        f"{total_combinacoes}"
    )

    print(
        f"Após filtros: "
        f"{total_apos_filtros}"
    )

    print(
        f"Jogos selecionados: "
        f"{len(jogos)}"
    )

    # ========================================================
    # 10. MOSTRAR JOGOS
    # ========================================================

    imprimir_jogos(
        jogos
    )

    # ========================================================
    # 11. CUSTO
    # ========================================================

    custo = calcular_custo(
        quantidade_jogos=
            len(jogos),

        custo_unitario=
            PRECO_APOSTA
    )

    print()
    print("=" * 65)
    print(
        "RESUMO"
    )
    print("=" * 65)

    print(
        f"Dezenas excluídas: "
        f"{len(exclusoes)}"
    )

    print(
        f"Dezenas candidatas: "
        f"{len(candidatas)}"
    )

    print(
        f"Universo completo: "
        f"{total_combinacoes} jogos"
    )

    print(
        f"Fechamento reduzido: "
        f"{len(jogos)} jogos"
    )

    print(
        f"Custo por aposta: "
        f"R$ {PRECO_APOSTA:.2f}"
    )

    print(
        f"Custo total estimado: "
        f"R$ {custo:.2f}"
    )

    print()
    print(
        "ATENÇÃO:"
    )

    print(
        "Este resultado ainda NÃO significa "
        "que o modelo possui poder preditivo."
    )

    print(
        "Precisamos validar a estratégia "
        "com backtesting walk-forward."
    )


if __name__ == "__main__":
    main()