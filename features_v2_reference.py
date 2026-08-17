import numpy as np


class GeradorFeaturesV2:
    """
    Dataset no formato:

        concurso + dezena -> features -> saiu no próximo concurso?

    Cada concurso gera 25 observações.

    Exemplo:

        concurso 3500 + dezena 01 -> features -> saiu/não saiu no 3501
        concurso 3500 + dezena 02 -> features -> saiu/não saiu no 3501
        ...
        concurso 3500 + dezena 25 -> features -> saiu/não saiu no 3501

    Esta é a versão simples/referência.
    Ela recalcula as estatísticas diretamente a partir do histórico,
    privilegiando correção e legibilidade.
    """

    def __init__(self, df_bolas):
        self.df_bolas = df_bolas.copy()

        self.total_sorteios = len(
            self.df_bolas
        )

        self.matriz_binaria = (
            self._criar_matriz_binaria()
        )

    # ========================================================
    # MATRIZ BINÁRIA
    # ========================================================

    def _criar_matriz_binaria(self):
        matriz = np.zeros(
            (
                self.total_sorteios,
                25
            ),
            dtype=np.int8
        )

        for indice, sorteio in enumerate(
            self.df_bolas.to_numpy()
        ):

            for bola in sorteio:
                bola = int(bola)

                if bola < 1 or bola > 25:
                    raise ValueError(
                        f"Dezena inválida encontrada: {bola}"
                    )

                matriz[
                    indice,
                    bola - 1
                ] = 1

        return matriz

    # ========================================================
    # FREQUÊNCIA
    # ========================================================

    @staticmethod
    def _frequencia(
        serie,
        janela
    ):
        if len(serie) == 0:
            return 0.0

        janela_real = min(
            janela,
            len(serie)
        )

        return float(
            np.mean(
                serie[
                    -janela_real:
                ]
            )
        )

    # ========================================================
    # ATRASO ATUAL
    # ========================================================

    @staticmethod
    def _calcular_atraso(
        serie
    ):
        atraso = 0

        for valor in reversed(
            serie
        ):

            if valor == 1:
                break

            atraso += 1

        return atraso

    # ========================================================
    # SEQUÊNCIA DE PRESENÇA
    # ========================================================

    @staticmethod
    def _sequencia_presenca(
        serie
    ):
        sequencia = 0

        for valor in reversed(
            serie
        ):

            if valor == 0:
                break

            sequencia += 1

        return sequencia

    # ========================================================
    # INTERVALOS DE AUSÊNCIA
    # ========================================================

    @staticmethod
    def _intervalos_ausencia(
        serie
    ):
        posicoes = np.where(
            serie == 1
        )[0]

        if len(posicoes) < 2:

            return np.array(
                [],
                dtype=np.int16
            )

        return (
            np.diff(
                posicoes
            )
            - 1
        )

    # ========================================================
    # PERCENTIL DO ATRASO
    # ========================================================

    @staticmethod
    def _percentil_atraso(
        atraso,
        intervalos
    ):
        if len(intervalos) == 0:
            return 0.0

        return float(
            np.mean(
                intervalos
                <= atraso
            )
        )

    # ========================================================
    # P(SAIR | ATRASO)
    # ========================================================

    @staticmethod
    def _prob_sair_dado_atraso(
        serie,
        atraso_desejado
    ):
        """
        Calcula:

            P(sair no próximo concurso |
              atraso atual = X)
        """

        if atraso_desejado <= 0:
            return 0.0, 0

        atraso_atual = 0

        amostras = 0
        sucessos = 0

        for indice in range(
            len(serie) - 1
        ):

            if serie[indice] == 1:
                atraso_atual = 0
            else:
                atraso_atual += 1

            if (
                atraso_atual
                == atraso_desejado
            ):
                amostras += 1

                if (
                    serie[
                        indice + 1
                    ]
                    == 1
                ):
                    sucessos += 1

        if amostras == 0:
            return 0.0, 0

        return (
            sucessos / amostras,
            amostras
        )

    # ========================================================
    # P(REPETIR | SEQUÊNCIA)
    # ========================================================

    @staticmethod
    def _prob_repetir_dado_sequencia(
        serie,
        sequencia_desejada
    ):
        if sequencia_desejada <= 0:
            return 0.0, 0

        sequencia_atual = 0

        amostras = 0
        sucessos = 0

        for indice in range(
            len(serie) - 1
        ):

            if serie[indice] == 1:
                sequencia_atual += 1
            else:
                sequencia_atual = 0

            if (
                sequencia_atual
                == sequencia_desejada
            ):
                amostras += 1

                if (
                    serie[
                        indice + 1
                    ]
                    == 1
                ):
                    sucessos += 1

        if amostras == 0:
            return 0.0, 0

        return (
            sucessos / amostras,
            amostras
        )

    # ========================================================
    # FEATURES DE UM CONCURSO
    # ========================================================

    def calcular_features_concurso(
        self,
        indice
    ):
        """
        Usa somente os concursos:

            0 ... indice

        para prever:

            indice + 1

        Retorna 25 linhas,
        uma para cada dezena.
        """

        if indice < 0:
            raise ValueError(
                "O índice não pode ser negativo."
            )

        if indice >= self.total_sorteios:
            raise ValueError(
                f"Índice {indice} fora da base."
            )

        passado = (
            self.matriz_binaria[
                :indice + 1
            ]
        )

        estatisticas = []

        # ====================================================
        # PRIMEIRA PASSAGEM
        # ====================================================

        for indice_dezena in range(
            25
        ):

            serie = (
                passado[
                    :,
                    indice_dezena
                ]
            )

            freq_5 = (
                self._frequencia(
                    serie,
                    5
                )
            )

            freq_10 = (
                self._frequencia(
                    serie,
                    10
                )
            )

            freq_20 = (
                self._frequencia(
                    serie,
                    20
                )
            )

            freq_50 = (
                self._frequencia(
                    serie,
                    50
                )
            )

            freq_100 = (
                self._frequencia(
                    serie,
                    100
                )
            )

            freq_200 = (
                self._frequencia(
                    serie,
                    200
                )
            )

            freq_historica = float(
                np.mean(
                    serie
                )
            )

            atraso = (
                self._calcular_atraso(
                    serie
                )
            )

            sequencia = (
                self._sequencia_presenca(
                    serie
                )
            )

            intervalos = (
                self._intervalos_ausencia(
                    serie
                )
            )

            if len(intervalos) > 0:

                media_atraso = float(
                    np.mean(
                        intervalos
                    )
                )

                mediana_atraso = float(
                    np.median(
                        intervalos
                    )
                )

                desvio_atraso = float(
                    np.std(
                        intervalos
                    )
                )

                max_atraso = int(
                    np.max(
                        intervalos
                    )
                )

            else:

                media_atraso = 0.0
                mediana_atraso = 0.0
                desvio_atraso = 0.0
                max_atraso = 0

            percentil_atraso = (
                self._percentil_atraso(
                    atraso,
                    intervalos
                )
            )

            (
                prob_sair_atraso,
                amostras_atraso
            ) = (
                self._prob_sair_dado_atraso(
                    serie,
                    atraso
                )
            )

            (
                prob_repetir,
                amostras_repeticao
            ) = (
                self._prob_repetir_dado_sequencia(
                    serie,
                    sequencia
                )
            )

            saiu_anterior = int(
                serie[-1]
            )

            estatisticas.append({
                "dezena":
                    indice_dezena + 1,

                "freq_5":
                    freq_5,

                "freq_10":
                    freq_10,

                "freq_20":
                    freq_20,

                "freq_50":
                    freq_50,

                "freq_100":
                    freq_100,

                "freq_200":
                    freq_200,

                "freq_historica":
                    freq_historica,

                "atraso":
                    atraso,

                "percentil_atraso":
                    percentil_atraso,

                "media_atraso":
                    media_atraso,

                "mediana_atraso":
                    mediana_atraso,

                "desvio_atraso":
                    desvio_atraso,

                "max_atraso":
                    max_atraso,

                "sequencia_presenca":
                    sequencia,

                "prob_sair_atraso":
                    prob_sair_atraso,

                "amostras_atraso":
                    amostras_atraso,

                "prob_repetir":
                    prob_repetir,

                "amostras_repeticao":
                    amostras_repeticao,

                "saiu_anterior":
                    saiu_anterior
            })

        # ====================================================
        # FEATURES RELATIVAS
        # ====================================================

        freq20 = np.asarray(
            [
                item["freq_20"]
                for item in estatisticas
            ],
            dtype=np.float32
        )

        atrasos = np.asarray(
            [
                item["atraso"]
                for item in estatisticas
            ],
            dtype=np.float32
        )

        media_freq20 = float(
            np.mean(
                freq20
            )
        )

        media_atrasos = float(
            np.mean(
                atrasos
            )
        )

        ranking_freq = (
            np.argsort(
                np.argsort(
                    -freq20
                )
            )
            + 1
        )

        ranking_atraso = (
            np.argsort(
                np.argsort(
                    -atrasos
                )
            )
            + 1
        )

        linhas = []

        # ====================================================
        # SEGUNDA PASSAGEM
        # ====================================================

        for indice_dezena, item in enumerate(
            estatisticas
        ):

            if (
                item[
                    "amostras_atraso"
                ]
                > 0
            ):

                lift_atraso = (
                    item[
                        "prob_sair_atraso"
                    ]
                    - 0.60
                )

            else:

                lift_atraso = 0.0

            if (
                item[
                    "amostras_repeticao"
                ]
                > 0
            ):

                lift_repeticao = (
                    item[
                        "prob_repetir"
                    ]
                    - 0.60
                )

            else:

                lift_repeticao = 0.0

            linha = [
                # 01
                item[
                    "dezena"
                ] / 25.0,

                # 02
                item[
                    "freq_5"
                ],

                # 03
                item[
                    "freq_10"
                ],

                # 04
                item[
                    "freq_20"
                ],

                # 05
                item[
                    "freq_50"
                ],

                # 06
                item[
                    "freq_100"
                ],

                # 07
                item[
                    "freq_200"
                ],

                # 08
                item[
                    "freq_historica"
                ],

                # 09
                item[
                    "atraso"
                ],

                # 10
                item[
                    "percentil_atraso"
                ],

                # 11
                item[
                    "media_atraso"
                ],

                # 12
                item[
                    "mediana_atraso"
                ],

                # 13
                item[
                    "desvio_atraso"
                ],

                # 14
                item[
                    "max_atraso"
                ],

                # 15
                item[
                    "sequencia_presenca"
                ],

                # 16
                item[
                    "prob_sair_atraso"
                ],

                # 17
                np.log1p(
                    item[
                        "amostras_atraso"
                    ]
                ),

                # 18
                item[
                    "prob_repetir"
                ],

                # 19
                np.log1p(
                    item[
                        "amostras_repeticao"
                    ]
                ),

                # 20
                lift_atraso,

                # 21
                lift_repeticao,

                # 22
                item[
                    "saiu_anterior"
                ],

                # 23
                item[
                    "freq_20"
                ]
                - media_freq20,

                # 24
                item[
                    "atraso"
                ]
                - media_atrasos,

                # 25
                ranking_freq[
                    indice_dezena
                ]
                / 25.0,

                # 26
                ranking_atraso[
                    indice_dezena
                ]
                / 25.0
            ]

            linhas.append(
                linha
            )

        return np.asarray(
            linhas,
            dtype=np.float32
        )

    # ========================================================
    # DATASET COMPLETO
    # ========================================================

    def construir_dataset(
        self,
        janela_minima=200
    ):
        """
        Dataset:

            X:
                aproximadamente 89 mil × 26

            y:
                uma classe por linha

                1 = saiu
                0 = não saiu

            indices_target:
                concurso que estamos prevendo

            dezenas:
                qual dezena aquela linha representa
        """

        if janela_minima < 1:
            raise ValueError(
                "janela_minima deve ser maior que zero."
            )

        if (
            janela_minima
            >= self.total_sorteios - 1
        ):
            raise ValueError(
                "janela_minima maior "
                "que histórico disponível."
            )

        X_blocos = []
        y_blocos = []

        indices_target = []
        dezenas = []

        for indice in range(
            janela_minima,
            self.total_sorteios - 1
        ):

            if (
                indice % 100 == 0
            ):
                print(
                    f"Gerando features V2: "
                    f"{indice}/"
                    f"{self.total_sorteios - 1}"
                )

            features = (
                self.calcular_features_concurso(
                    indice
                )
            )

            target = (
                self.matriz_binaria[
                    indice + 1
                ]
            )

            X_blocos.append(
                features
            )

            y_blocos.append(
                target
            )

            indices_target.extend(
                [
                    indice + 1
                ]
                * 25
            )

            dezenas.extend(
                range(
                    1,
                    26
                )
            )

        X = np.vstack(
            X_blocos
        )

        y = np.concatenate(
            y_blocos
        )

        return (
            np.asarray(
                X,
                dtype=np.float32
            ),

            np.asarray(
                y,
                dtype=np.int8
            ),

            np.asarray(
                indices_target,
                dtype=np.int32
            ),

            np.asarray(
                dezenas,
                dtype=np.int8
            )
        )

    # ========================================================
    # PRÓXIMO CONCURSO
    # ========================================================

    def features_proximo_concurso(
        self
    ):
        ultimo_indice = (
            self.total_sorteios
            - 1
        )

        return (
            self.calcular_features_concurso(
                ultimo_indice
            )
        )

    # ========================================================
    # NOMES
    # ========================================================

    @staticmethod
    def nomes_features():
        return [
            "dezena_normalizada",

            "freq_5",
            "freq_10",
            "freq_20",
            "freq_50",
            "freq_100",
            "freq_200",
            "freq_historica",

            "atraso",
            "percentil_atraso",
            "media_atraso",
            "mediana_atraso",
            "desvio_atraso",
            "max_atraso",

            "sequencia_presenca",

            "prob_sair_atraso",
            "log_amostras_atraso",

            "prob_repetir",
            "log_amostras_repeticao",

            "lift_atraso",
            "lift_repeticao",

            "saiu_anterior",

            "freq_20_relativa",
            "atraso_relativo",

            "ranking_freq_20",
            "ranking_atraso"
        ]

    @staticmethod
    def quantidade_features():
        return 26