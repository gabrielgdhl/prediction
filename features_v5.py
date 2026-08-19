import numpy as np

from config_v5 import (
    JANELAS_FREQUENCIA_V5,
    PARES_TENDENCIA_V5,
)


# ============================================================
# FREQUÊNCIA
# ============================================================

def calcular_frequencia(
    serie,
    janela
):
    """
    Frequência da dezena nos últimos N concursos.

    Exemplo:

        serie final = [1, 0, 1]

        janela = 3

        freq = 2 / 3
    """

    tamanho = min(
        janela,
        len(serie)
    )

    if tamanho == 0:
        return 0.0

    return float(
        np.mean(
            serie[-tamanho:]
        )
    )


# ============================================================
# SEQUÊNCIA ATUAL
# ============================================================

def calcular_sequencia_atual(
    serie
):
    """
    Quantos concursos consecutivos a dezena
    está presente neste momento.

    ... 0 1 1 1

    sequência = 3
    """

    sequencia = 0

    for valor in reversed(
        serie
    ):

        if valor == 0:
            break

        sequencia += 1

    return sequencia


# ============================================================
# SEQUÊNCIAS HISTÓRICAS
# ============================================================

def extrair_sequencias_finalizadas(
    serie
):
    """
    Extrai sequências de presença que sabemos
    que efetivamente terminaram.

    Exemplo:

        0 1 1 0 1 1 1 0

    resultado:

        [2, 3]

    IMPORTANTE:

    Se a série terminar:

        0 1 1

    a sequência final NÃO entra.

    Ela ainda está acontecendo e não sabemos
    seu tamanho final.

    Isso evita leakage.
    """

    sequencias = []

    atual = 0

    for valor in serie:

        if valor == 1:

            atual += 1

        else:

            if atual > 0:

                sequencias.append(
                    atual
                )

                atual = 0

    return sequencias


# ============================================================
# SOBREVIVÊNCIA DA SEQUÊNCIA
# ============================================================

def calcular_sobrevivencia_sequencia(
    serie
):
    """
    Mede a probabilidade histórica de uma sequência
    continuar dado seu tamanho atual.

    Exemplo:

        sequência atual = 1

    Procuramos historicamente:

        quantas sequências chegaram a >= 1

    e:

        quantas chegaram a >= 2

    Então:

        P(continuar | sequência >= 1)

    Essa feature representa exatamente a hipótese:

        "essa dezena normalmente não aparece apenas
         uma vez quando inicia uma sequência?"
    """

    serie = np.asarray(
        serie,
        dtype=np.int8
    )

    sequencia_atual = (
        calcular_sequencia_atual(
            serie
        )
    )

    baseline_presenca = float(
        np.mean(
            serie
        )
    )

    # --------------------------------------------------------
    # Dezena não está atualmente em sequência
    # --------------------------------------------------------

    if sequencia_atual == 0:

        return {
            "sequencia_atual_v5":
                0,

            "prob_sobreviver_sequencia":
                baseline_presenca,

            "lift_sobrevivencia":
                0.0,

            "amostras_sobrevivencia":
                0,

            "prob_terminar_agora":
                1.0 - baseline_presenca,
        }

    sequencias = (
        extrair_sequencias_finalizadas(
            serie
        )
    )

    # --------------------------------------------------------
    # Quantas sequências históricas chegaram
    # pelo menos ao tamanho atual?
    # --------------------------------------------------------

    amostras = sum(
        tamanho >= sequencia_atual
        for tamanho
        in sequencias
    )

    # --------------------------------------------------------
    # Dessas, quantas continuaram pelo menos mais uma vez?
    # --------------------------------------------------------

    sucessos = sum(
        tamanho >= (
            sequencia_atual + 1
        )
        for tamanho
        in sequencias
    )

    if amostras == 0:

        prob_sobreviver = (
            baseline_presenca
        )

    else:

        prob_sobreviver = (
            sucessos
            / amostras
        )

    return {
        "sequencia_atual_v5":
            sequencia_atual,

        "prob_sobreviver_sequencia":
            float(
                prob_sobreviver
            ),

        "lift_sobrevivencia":
            float(
                prob_sobreviver
                - baseline_presenca
            ),

        "amostras_sobrevivencia":
            int(
                amostras
            ),

        "prob_terminar_agora":
            float(
                1.0
                - prob_sobreviver
            ),
    }


# ============================================================
# RANK
# ============================================================

def calcular_ranks_desc(
    valores
):
    """
    Maior valor recebe rank 1.

    Retorna ranks de 1 a 25.
    """

    valores = np.asarray(
        valores,
        dtype=float
    )

    ordem = np.argsort(
        -valores
    )

    ranks = np.empty(
        len(valores),
        dtype=np.int16
    )

    ranks[
        ordem
    ] = (
        np.arange(
            len(valores)
        )
        + 1
    )

    return ranks


# ============================================================
# NORMALIZAÇÃO DE RANK
# ============================================================

def normalizar_rank(
    rank,
    total=25
):
    """
    rank 1  -> 0
    rank 25 -> 1

    Mantemos assim porque representa distância
    do topo do ranking.
    """

    if total <= 1:
        return 0.0

    return (
        (rank - 1)
        / (total - 1)
    )


def calcular_percentil_rank(
    rank,
    total=25
):
    """
    rank 1  -> 1
    rank 25 -> 0
    """

    return (
        1.0
        - normalizar_rank(
            rank,
            total
        )
    )


# ============================================================
# FEATURES DE UM CONCURSO
# ============================================================

def calcular_features_v5_concurso(
    matriz_binaria,
    indice_estado
):
    """
    Calcula as features extras V5 para as 25 dezenas.

    indice_estado representa o último concurso
    CONHECIDO.

    Exemplo:

        queremos prever concurso 3000

        indice_estado = 2999

    Portanto:

        matriz_binaria[:3000]

    e nunca:

        matriz_binaria[:3001]

    Isso é essencial para impedir leakage.
    """

    if indice_estado < 0:

        raise ValueError(
            "indice_estado não pode ser negativo."
        )

    if indice_estado >= len(
        matriz_binaria
    ):

        raise ValueError(
            "indice_estado ultrapassa "
            "matriz_binaria."
        )

    passado = np.asarray(
        matriz_binaria[
            :indice_estado + 1
        ],
        dtype=np.int8
    )

    if passado.shape[1] != 25:

        raise ValueError(
            "matriz_binaria deve possuir "
            "25 colunas."
        )

    extras = {
        dezena: {}
        for dezena
        in range(
            1,
            26
        )
    }

    # ========================================================
    # FREQUÊNCIAS
    # ========================================================

    frequencias = {}

    for janela in (
        JANELAS_FREQUENCIA_V5
    ):

        valores = np.zeros(
            25,
            dtype=float
        )

        for indice_dezena in range(
            25
        ):

            serie = passado[
                :,
                indice_dezena
            ]

            valor = (
                calcular_frequencia(
                    serie,
                    janela
                )
            )

            valores[
                indice_dezena
            ] = valor

            dezena = (
                indice_dezena
                + 1
            )

            extras[
                dezena
            ][
                f"freq_{janela}"
            ] = valor

        frequencias[
            janela
        ] = valores

        # ----------------------------------------------------
        # Contexto relativo das 25 dezenas
        # ----------------------------------------------------

        media = float(
            np.mean(
                valores
            )
        )

        ranks = (
            calcular_ranks_desc(
                valores
            )
        )

        for indice_dezena in range(
            25
        ):

            dezena = (
                indice_dezena
                + 1
            )

            rank = int(
                ranks[
                    indice_dezena
                ]
            )

            extras[
                dezena
            ][
                f"rank_freq_{janela}"
            ] = (
                normalizar_rank(
                    rank
                )
            )

            extras[
                dezena
            ][
                f"percentil_freq_{janela}"
            ] = (
                calcular_percentil_rank(
                    rank
                )
            )

            extras[
                dezena
            ][
                f"freq_{janela}_relativa"
            ] = float(
                valores[
                    indice_dezena
                ]
                - media
            )

    # ========================================================
    # TENDÊNCIAS ENTRE JANELAS
    # ========================================================

    for (
        janela_curta,
        janela_longa
    ) in PARES_TENDENCIA_V5:

        valores_curta = (
            frequencias[
                janela_curta
            ]
        )

        valores_longa = (
            frequencias[
                janela_longa
            ]
        )

        tendencia = (
            valores_curta
            - valores_longa
        )

        ranks = (
            calcular_ranks_desc(
                tendencia
            )
        )

        for indice_dezena in range(
            25
        ):

            dezena = (
                indice_dezena
                + 1
            )

            valor = float(
                tendencia[
                    indice_dezena
                ]
            )

            rank = int(
                ranks[
                    indice_dezena
                ]
            )

            extras[
                dezena
            ][
                (
                    f"tendencia_"
                    f"{janela_curta}_"
                    f"{janela_longa}"
                )
            ] = valor

            extras[
                dezena
            ][
                (
                    f"rank_tendencia_"
                    f"{janela_curta}_"
                    f"{janela_longa}"
                )
            ] = (
                normalizar_rank(
                    rank
                )
            )

    # ========================================================
    # SOBREVIVÊNCIA DAS SEQUÊNCIAS
    # ========================================================

    probs_sobrevivencia = np.zeros(
        25,
        dtype=float
    )

    for indice_dezena in range(
        25
    ):

        dezena = (
            indice_dezena
            + 1
        )

        serie = passado[
            :,
            indice_dezena
        ]

        sobrevivencia = (
            calcular_sobrevivencia_sequencia(
                serie
            )
        )

        extras[
            dezena
        ].update(
            sobrevivencia
        )

        # log1p evita deixar número de amostras
        # em escala completamente diferente.
        extras[
            dezena
        ][
            "log_amostras_sobrevivencia"
        ] = float(
            np.log1p(
                sobrevivencia[
                    "amostras_sobrevivencia"
                ]
            )
        )

        probs_sobrevivencia[
            indice_dezena
        ] = (
            sobrevivencia[
                "prob_sobreviver_sequencia"
            ]
        )

    # ========================================================
    # RANK DE SOBREVIVÊNCIA
    # ========================================================

    ranks_sobrevivencia = (
        calcular_ranks_desc(
            probs_sobrevivencia
        )
    )

    media_sobrevivencia = float(
        np.mean(
            probs_sobrevivencia
        )
    )

    for indice_dezena in range(
        25
    ):

        dezena = (
            indice_dezena
            + 1
        )

        rank = int(
            ranks_sobrevivencia[
                indice_dezena
            ]
        )

        extras[
            dezena
        ][
            "rank_sobrevivencia"
        ] = (
            normalizar_rank(
                rank
            )
        )

        extras[
            dezena
        ][
            "percentil_sobrevivencia"
        ] = (
            calcular_percentil_rank(
                rank
            )
        )

        extras[
            dezena
        ][
            "sobrevivencia_relativa"
        ] = float(
            probs_sobrevivencia[
                indice_dezena
            ]
            - media_sobrevivencia
        )

    return extras