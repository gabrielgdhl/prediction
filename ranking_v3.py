import math


# ============================================================
# CONFIGURAÇÕES INICIAIS DA V3
# ============================================================

# Top 1 e Top 2 do ranking de exclusão são soberanos.
TOP_EXCLUSAO_SOBERANO = 2


# ------------------------------------------------------------
# CONFLITO:
#
# presença estatística precisa superar a evidência de
# exclusão por pelo menos esta margem.
# ------------------------------------------------------------

MARGEM_MINIMA_CONFLITO = 0.03


# ------------------------------------------------------------
# EXAUSTÃO
#
# Não usamos mais "85% de exaustão".
#
# Agora procuramos aumento da probabilidade de parar
# em relação ao baseline histórico da própria dezena.
# ------------------------------------------------------------

LIFT_EXAUSTAO_MINIMO = 0.04

AMOSTRAS_EXAUSTAO_MINIMAS = 30


# ------------------------------------------------------------
# SHRINKAGE
#
# Evita acreditar demais em probabilidades calculadas
# com poucas amostras.
# ------------------------------------------------------------

FORCA_PRIOR = 50


# ============================================================
# UTILITÁRIOS
# ============================================================

def recuperar_amostras(log_amostras):
    """
    features_v2 armazena:

        log1p(amostras)

    Recuperamos o número aproximadamente original.
    """

    return max(
        0,
        int(
            round(
                math.expm1(
                    float(log_amostras)
                )
            )
        )
    )


def aplicar_shrinkage(
    probabilidade,
    baseline,
    amostras,
    forca_prior=FORCA_PRIOR
):
    """
    Aproxima uma probabilidade de baixa amostra
    do baseline.

    Exemplo:

        estimativa = 70%
        baseline   = 60%
        amostras   = 5

    não queremos tratar 70% como se viesse
    de centenas de observações.
    """

    if amostras <= 0:
        return float(baseline)

    peso = (
        amostras
        / (
            amostras
            + forca_prior
        )
    )

    return float(
        baseline
        + (
            probabilidade
            - baseline
        )
        * peso
    )


def wilson_inferior(
    probabilidade,
    amostras,
    z=1.96
):
    """
    Limite inferior do intervalo de confiança
    de Wilson para uma proporção.
    """

    if amostras <= 0:
        return 0.0

    p = float(probabilidade)
    n = int(amostras)

    denominador = (
        1
        + (
            z ** 2
            / n
        )
    )

    centro = (
        p
        + (
            z ** 2
            / (2 * n)
        )
    )

    margem = (
        z
        * math.sqrt(
            (
                p * (1 - p)
                + (
                    z ** 2
                    / (4 * n)
                )
            )
            / n
        )
    )

    return max(
        0.0,
        (
            centro
            - margem
        )
        / denominador
    )


# ============================================================
# SCORE ESTATÍSTICO DE PRESENÇA
# ============================================================

def calcular_score_presenca(
    features
):
    """
    Calcula a evidência estatística de que a dezena
    apareça no próximo concurso.

    Se ela está saindo em sequência:
        usa P(repetir | sequência)

    Se ela está ausente:
        usa P(sair | atraso)

    A probabilidade histórica da própria dezena
    funciona como baseline.
    """

    baseline_presenca = float(
        features[
            "freq_historica"
        ]
    )

    atraso = int(
        features[
            "atraso"
        ]
    )

    sequencia = int(
        features[
            "sequencia_presenca"
        ]
    )

    # ========================================================
    # ESTÁ EM SEQUÊNCIA DE PRESENÇA
    # ========================================================

    if sequencia > 0:

        prob_condicional = float(
            features[
                "prob_repetir"
            ]
        )

        amostras = recuperar_amostras(
            features[
                "log_amostras_repeticao"
            ]
        )

        tipo = "REPETICAO"

    # ========================================================
    # ESTÁ AUSENTE
    # ========================================================

    elif atraso > 0:

        prob_condicional = float(
            features[
                "prob_sair_atraso"
            ]
        )

        amostras = recuperar_amostras(
            features[
                "log_amostras_atraso"
            ]
        )

        tipo = "RETORNO"

    else:

        prob_condicional = (
            baseline_presenca
        )

        amostras = 0

        tipo = "BASELINE"

    score = aplicar_shrinkage(
        probabilidade=
            prob_condicional,

        baseline=
            baseline_presenca,

        amostras=
            amostras
    )

    lift = (
        score
        - baseline_presenca
    )

    return {
        "score_presenca":
            score,

        "prob_condicional":
            prob_condicional,

        "baseline_presenca":
            baseline_presenca,

        "lift_presenca":
            lift,

        "amostras":
            amostras,

        "tipo_sinal":
            tipo
    }


# ============================================================
# RANKING ESTATÍSTICO DE PROVÁVEIS
# ============================================================

def obter_provaveis_estatisticos(
    features_por_dezena,
    quantidade=5
):
    """
    Esta é a mudança principal da V3.

    NÃO usamos mais o Random Forest V2 para criar
    o conjunto de prováveis.

    Usamos apenas as estatísticas individuais
    de repetição/retorno.
    """

    ranking = []

    for dezena, features in (
        features_por_dezena.items()
    ):

        resultado = (
            calcular_score_presenca(
                features
            )
        )

        ranking.append({
            "dezena":
                int(dezena),

            **resultado
        })

    ranking.sort(
        key=lambda item: (
            item[
                "score_presenca"
            ],
            item[
                "amostras"
            ]
        ),
        reverse=True
    )

    top = ranking[
        :quantidade
    ]

    provaveis = {
        item["dezena"]:
            item["score_presenca"]
        for item in top
    }

    return (
        provaveis,
        ranking
    )


# ============================================================
# EXAUSTÃO
# ============================================================

def analisar_exaustao(
    features,
    lift_minimo=LIFT_EXAUSTAO_MINIMO,
    amostras_minimas=AMOSTRAS_EXAUSTAO_MINIMAS
):
    """
    Exemplo:

        frequência histórica da dezena = 60%

        baseline de NÃO sair = 40%

        sequência atual = 3

        P(repetir | seq=3) = 53%

        P(parar | seq=3) = 47%

        lift exaustão:

            47% - 40%
            = +7pp

    Isso é muito mais útil do que exigir
    arbitrariamente 85% de exaustão.
    """

    sequencia = int(
        features[
            "sequencia_presenca"
        ]
    )

    baseline_presenca = float(
        features[
            "freq_historica"
        ]
    )

    baseline_exaustao = (
        1.0
        - baseline_presenca
    )

    if sequencia <= 0:

        return {
            "exaustao":
                False,

            "sequencia":
                0,

            "prob_exaustao":
                baseline_exaustao,

            "prob_exaustao_ajustada":
                baseline_exaustao,

            "baseline_exaustao":
                baseline_exaustao,

            "lift_exaustao":
                0.0,

            "amostras":
                0,

            "ic_inferior":
                0.0
        }

    prob_repetir = float(
        features[
            "prob_repetir"
        ]
    )

    amostras = recuperar_amostras(
        features[
            "log_amostras_repeticao"
        ]
    )

    if amostras <= 0:

        return {
            "exaustao":
                False,

            "sequencia":
                sequencia,

            "prob_exaustao":
                baseline_exaustao,

            "prob_exaustao_ajustada":
                baseline_exaustao,

            "baseline_exaustao":
                baseline_exaustao,

            "lift_exaustao":
                0.0,

            "amostras":
                0,

            "ic_inferior":
                0.0
        }

    prob_exaustao = (
        1.0
        - prob_repetir
    )

    prob_ajustada = aplicar_shrinkage(
        probabilidade=
            prob_exaustao,

        baseline=
            baseline_exaustao,

        amostras=
            amostras
    )

    lift = (
        prob_ajustada
        - baseline_exaustao
    )

    ic_inferior = (
        wilson_inferior(
            probabilidade=
                prob_exaustao,

            amostras=
                amostras
        )
    )

    # ========================================================
    # REGRA DE EXAUSTÃO FORTE
    # ========================================================

    forte = (
        amostras
        >= amostras_minimas

        and

        lift
        >= lift_minimo

        and

        ic_inferior
        > baseline_exaustao
    )

    return {
        "exaustao":
            forte,

        "sequencia":
            sequencia,

        "prob_exaustao":
            prob_exaustao,

        "prob_exaustao_ajustada":
            prob_ajustada,

        "baseline_exaustao":
            baseline_exaustao,

        "lift_exaustao":
            lift,

        "amostras":
            amostras,

        "ic_inferior":
            ic_inferior
    }


# ============================================================
# SELEÇÃO HÍBRIDA V3
# ============================================================

def selecionar_exclusoes_v3(
    ranking_exclusao,
    provaveis,
    features_por_dezena,
    quantidade_exclusoes,

    top_soberano=
        TOP_EXCLUSAO_SOBERANO,

    margem_minima=
        MARGEM_MINIMA_CONFLITO,

    lift_exaustao_minimo=
        LIFT_EXAUSTAO_MINIMO,

    amostras_exaustao_minimas=
        AMOSTRAS_EXAUSTAO_MINIMAS
):
    """
    PRIORIDADE:

    1. Top-2 do ranking V2 é soberano.
    2. Exaustões estatisticamente fortes.
    3. Ranking normal.
    4. Conflitos abaixo do Top-2 são arbitrados
       com o ranking estatístico de presença.
    """

    ranking_posicao = {
        item["dezena"]:
            posicao

        for posicao, item
        in enumerate(
            ranking_exclusao,
            start=1
        )
    }

    ranking_por_dezena = {
        item["dezena"]:
            item

        for item
        in ranking_exclusao
    }

    exclusoes = []

    detalhes = []

    # ========================================================
    # 1. TOP-2 SOBERANO
    # ========================================================

    for item in ranking_exclusao[
        :top_soberano
    ]:

        dezena = int(
            item[
                "dezena"
            ]
        )

        exclusoes.append(
            dezena
        )

        detalhes.append({
            "dezena":
                dezena,

            "decisao":
                "EXCLUIR",

            "motivo":
                "TOP_EXCLUSAO_SOBERANO",

            "ranking_exclusao":
                ranking_posicao[
                    dezena
                ],

            "prob_nao_sair":
                float(
                    item[
                        "prob_nao_sair"
                    ]
                )
        })

    # ========================================================
    # 2. PROCURAR EXAUSTÕES
    # ========================================================

    candidatos_exaustao = []

    for dezena, features in (
        features_por_dezena.items()
    ):

        if dezena in exclusoes:
            continue

        resultado = analisar_exaustao(
            features,

            lift_minimo=
                lift_exaustao_minimo,

            amostras_minimas=
                amostras_exaustao_minimas
        )

        if resultado[
            "exaustao"
        ]:

            candidatos_exaustao.append({
                "dezena":
                    dezena,

                **resultado
            })

    candidatos_exaustao.sort(
        key=lambda item: (
            item[
                "lift_exaustao"
            ],
            item[
                "amostras"
            ]
        ),
        reverse=True
    )

    for candidato in (
        candidatos_exaustao
    ):

        if (
            len(exclusoes)
            >= quantidade_exclusoes
        ):
            break

        dezena = int(
            candidato[
                "dezena"
            ]
        )

        exclusoes.append(
            dezena
        )

        ranking_item = (
            ranking_por_dezena[
                dezena
            ]
        )

        detalhes.append({
            "dezena":
                dezena,

            "decisao":
                "EXCLUIR",

            "motivo":
                "EXAUSTAO_FORTE",

            "ranking_exclusao":
                ranking_posicao[
                    dezena
                ],

            "prob_nao_sair":
                float(
                    ranking_item[
                        "prob_nao_sair"
                    ]
                ),

            "prob_exaustao":
                candidato[
                    "prob_exaustao"
                ],

            "prob_exaustao_ajustada":
                candidato[
                    "prob_exaustao_ajustada"
                ],

            "baseline_exaustao":
                candidato[
                    "baseline_exaustao"
                ],

            "lift_exaustao":
                candidato[
                    "lift_exaustao"
                ],

            "sequencia":
                candidato[
                    "sequencia"
                ],

            "amostras_exaustao":
                candidato[
                    "amostras"
                ]
        })

    # ========================================================
    # 3. CONTINUAR ANDANDO O RANKING
    # ========================================================

    for item in ranking_exclusao:

        if (
            len(exclusoes)
            >= quantidade_exclusoes
        ):
            break

        dezena = int(
            item[
                "dezena"
            ]
        )

        if dezena in exclusoes:
            continue

        prob_nao_sair = float(
            item[
                "prob_nao_sair"
            ]
        )

        # ====================================================
        # SEM CONFLITO
        # ====================================================

        if dezena not in provaveis:

            exclusoes.append(
                dezena
            )

            detalhes.append({
                "dezena":
                    dezena,

                "decisao":
                    "EXCLUIR",

                "motivo":
                    "RANKING_EXCLUSAO",

                "ranking_exclusao":
                    ranking_posicao[
                        dezena
                    ],

                "prob_nao_sair":
                    prob_nao_sair
            })

            continue

        # ====================================================
        # CONFLITO
        #
        # Está simultaneamente:
        #
        # - no ranking de exclusão
        # - no conjunto estatístico de prováveis
        # ====================================================

        prob_presenca = float(
            provaveis[
                dezena
            ]
        )

        vantagem_presenca = (
            prob_presenca
            - prob_nao_sair
        )

        # ====================================================
        # PRESENÇA VENCE
        # ====================================================

        if (
            vantagem_presenca
            >= margem_minima
        ):

            detalhes.append({
                "dezena":
                    dezena,

                "decisao":
                    "PROTEGER",

                "motivo":
                    "PRESENCA_ESTATISTICA_VENCE",

                "ranking_exclusao":
                    ranking_posicao[
                        dezena
                    ],

                "prob_nao_sair":
                    prob_nao_sair,

                "score_presenca":
                    prob_presenca,

                "vantagem_presenca":
                    vantagem_presenca
            })

            # Não adiciona.
            #
            # O loop continua andando o ranking.
            continue

        # ====================================================
        # EXCLUSÃO VENCE / EMPATE
        # ====================================================

        exclusoes.append(
            dezena
        )

        detalhes.append({
            "dezena":
                dezena,

            "decisao":
                "EXCLUIR",

            "motivo":
                "EXCLUSAO_VENCE_CONFLITO",

            "ranking_exclusao":
                ranking_posicao[
                    dezena
                ],

            "prob_nao_sair":
                prob_nao_sair,

            "score_presenca":
                prob_presenca,

            "vantagem_presenca":
                vantagem_presenca
        })

    return (
        sorted(
            exclusoes
        ),
        detalhes
    )