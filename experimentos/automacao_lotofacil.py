import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import warnings
import time

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "Lotofácil.xlsx"

N_EXCLUSOES = 7

JANELA_FREQUENCIA = 15

N_CONCURSOS_BACKTEST = 200

N_ESTIMADORES = 100

MAX_DEPTH = 6

SEED = 42


# ============================================================
# CARREGAR LOTOFÁCIL
# ============================================================

def carregar_lotofacil():

    print("Carregando base...")

    df = pd.read_excel(ARQUIVO)

    # Procuramos automaticamente as colunas Bola1...Bola15
    colunas_bolas = [
        col
        for col in df.columns
        if "bola" in str(col).lower()
    ][:15]

    if len(colunas_bolas) != 15:

        print(
            "Não encontrei 15 colunas Bola."
        )

        print(
            "Colunas encontradas:",
            list(df.columns)
        )

        raise ValueError(
            "Não foi possível identificar as 15 dezenas."
        )

    df_bolas = (
        df[colunas_bolas]
        .dropna()
        .astype(int)
    )

    matriz = np.zeros(
        (len(df_bolas), 25),
        dtype=np.int8
    )

    for i, row in enumerate(
        df_bolas.to_numpy()
    ):

        for dezena in row:

            if not 1 <= dezena <= 25:
                raise ValueError(
                    f"Dezena inválida: {dezena}"
                )

            matriz[i, dezena - 1] = 1

    print(
        f"Base carregada: "
        f"{len(matriz)} concursos"
    )

    return df, matriz


# ============================================================
# FEATURES
# ============================================================

def criar_features(
    matriz,
    indice,
    janela=15
):

    historico = matriz[
        indice - janela:indice
    ]

    # Frequência de cada dezena
    frequencia = historico.sum(axis=0)

    return frequencia


# ============================================================
# TREINAR MODELO
# ============================================================

def treinar_modelo(
    matriz,
    indice
):

    X = []
    y = []

    # Precisamos de histórico suficiente
    inicio = JANELA_FREQUENCIA

    for i in range(
        inicio,
        indice
    ):

        features = criar_features(
            matriz,
            i,
            JANELA_FREQUENCIA
        )

        X.append(features)

        # Resultado do concurso i
        y.append(
            matriz[i]
        )

    X = np.asarray(X)
    y = np.asarray(y)

    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMADORES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    modelo.fit(
        X,
        y
    )

    return modelo


# ============================================================
# PROBABILIDADES
# ============================================================

def obter_probabilidades_nao_sair(
    modelo,
    features
):

    features = np.asarray(
        features
    ).reshape(1, -1)

    probabilidades = (
        modelo.predict_proba(
            features
        )
    )

    resultado = []

    for dezena in range(25):

        classes = modelo.classes_[dezena]

        probs = probabilidades[dezena][0]

        # ------------------------------------------
        # Encontrar P(classe = 0)
        #
        # classe 0 = não saiu
        # classe 1 = saiu
        # ------------------------------------------

        if 0 in classes:

            indice_classe_0 = np.where(
                classes == 0
            )[0][0]

            prob_nao_sair = probs[
                indice_classe_0
            ]

        else:

            # Se o modelo nunca viu classe 0,
            # significa que todos os exemplos
            # daquele target foram 1.
            prob_nao_sair = 0.0

        resultado.append({

            "dezena": dezena + 1,

            "prob_nao_sair":
                float(prob_nao_sair)
        })

    resultado = sorted(
        resultado,
        key=lambda x: x["prob_nao_sair"],
        reverse=True
    )

    return resultado


# ============================================================
# SELECIONAR EXCLUSÕES
# ============================================================

def selecionar_exclusoes(
    ranking,
    quantidade=7
):

    exclusoes = sorted([
        item["dezena"]
        for item in ranking[:quantidade]
    ])

    return exclusoes


# ============================================================
# BACKTEST
# ============================================================

def executar_backtest(
    matriz
):

    total = len(matriz)

    inicio = max(
        JANELA_FREQUENCIA + 1,
        total - N_CONCURSOS_BACKTEST
    )

    rng = np.random.default_rng(
        SEED
    )

    resultados = []

    start_time = time.time()

    print()
    print(
        f"Executando backtest: "
        f"concursos {inicio} até {total - 1}"
    )

    print()

    for idx in range(
        inicio,
        total
    ):

        # ==========================================
        # TREINAMENTO
        # ==========================================

        modelo = treinar_modelo(
            matriz,
            idx
        )

        # ==========================================
        # FEATURES DO CONCURSO ATUAL
        # ==========================================

        features_atuais = criar_features(
            matriz,
            idx,
            JANELA_FREQUENCIA
        )

        # ==========================================
        # PROBABILIDADES
        # ==========================================

        ranking = obter_probabilidades_nao_sair(
            modelo,
            features_atuais
        )

        # ==========================================
        # 7 EXCLUSÕES
        # ==========================================

        exclusoes_ml = selecionar_exclusoes(
            ranking,
            N_EXCLUSOES
        )

        exclusoes_ml = set(
            exclusoes_ml
        )

        # ==========================================
        # BASELINE ALEATÓRIO
        # ==========================================

        exclusoes_random = set(
            rng.choice(
                np.arange(1, 26),
                size=N_EXCLUSOES,
                replace=False
            )
        )

        # ==========================================
        # RESULTADO REAL
        # ==========================================

        sorteadas = set(
            np.where(
                matriz[idx] == 1
            )[0] + 1
        )

        nao_sorteadas = (
            set(range(1, 26))
            - sorteadas
        )

        # ==========================================
        # ACERTOS DE EXCLUSÃO
        # ==========================================

        acertos_ml = len(
            exclusoes_ml
            & nao_sorteadas
        )

        acertos_random = len(
            exclusoes_random
            & nao_sorteadas
        )

        # ==========================================
        # DEZENAS CANDIDATAS
        # ==========================================

        candidatas_ml = sorted(
            set(range(1, 26))
            - exclusoes_ml
        )

        # ==========================================
        # SALVAR RESULTADO
        # ==========================================

        resultados.append({

            "indice": idx,

            "acertos_ml":
                acertos_ml,

            "acertos_random":
                acertos_random,

            "exclusoes_ml":
                " ".join(
                    f"{d:02d}"
                    for d in sorted(
                        exclusoes_ml
                    )
                ),

            "exclusoes_reais":
                " ".join(
                    f"{d:02d}"
                    for d in sorted(
                        nao_sorteadas
                    )
                ),

            "candidatas_ml":
                " ".join(
                    f"{d:02d}"
                    for d in candidatas_ml
                ),

            "melhor_probabilidade":
                ranking[0]["prob_nao_sair"],

            "pior_probabilidade":
                ranking[-1]["prob_nao_sair"]
        })

        # ==========================================
        # PROGRESSO
        # ==========================================

        processados = (
            idx - inicio + 1
        )

        if (
            processados % 25 == 0
            or idx == inicio
        ):

            print(
                f"Processados: "
                f"{processados}/"
                f"{total - inicio}"
            )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# ANALISAR RESULTADOS
# ============================================================

def analisar_resultados(
    resultados
):

    media_ml = resultados[
        "acertos_ml"
    ].mean()

    media_random = resultados[
        "acertos_random"
    ].mean()

    ganho = (
        (media_ml / media_random) - 1
    ) * 100

    print()
    print("=" * 60)
    print("RESULTADO DO BACKTEST")
    print("=" * 60)

    print(
        f"Concursos analisados: "
        f"{len(resultados)}"
    )

    print()

    print(
        f"ML:       {media_ml:.3f}"
    )

    print(
        f"Aleatório:{media_random:.3f}"
    )

    print(
        f"Esperado: 2.800"
    )

    print()

    print(
        f"Ganho ML vs aleatório: "
        f"{ganho:.2f}%"
    )

    print()

    # Distribuição
    print(
        "Distribuição dos acertos ML:"
    )

    distribuicao = (
        resultados[
            "acertos_ml"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        distribuicao.to_string()
    )

    print()

    print(
        "Distribuição do aleatório:"
    )

    distribuicao_random = (
        resultados[
            "acertos_random"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        distribuicao_random.to_string()
    )


# ============================================================
# EXPORTAR
# ============================================================

def exportar_resultados(
    resultados
):

    arquivo_saida = (
        "backtest_lotofacil.xlsx"
    )

    resultados.to_excel(
        arquivo_saida,
        index=False
    )

    print()
    print(
        f"Resultado salvo em: "
        f"{arquivo_saida}"
    )


# ============================================================
# MAIN
# ============================================================

def rodar_analise_completa():

    print(
        "=============================================="
    )

    print(
        "LOTOFÁCIL - MODELO DE EXCLUSÃO"
    )

    print(
        "=============================================="
    )

    inicio_execucao = time.time()

    try:

        df, matriz = (
            carregar_lotofacil()
        )

    except Exception as e:

        print(
            f"Erro ao carregar dados: {e}"
        )

        return

    resultados = executar_backtest(
        matriz
    )

    analisar_resultados(
        resultados
    )

    exportar_resultados(
        resultados
    )

    print()

    print(
        f"Tempo total: "
        f"{time.time() - inicio_execucao:.1f}s"
    )


if __name__ == "__main__":

    rodar_analise_completa()