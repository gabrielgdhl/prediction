from collections import Counter, defaultdict

import numpy as np


class GeradorFeaturesV2:
    """
    V2 otimizada e validável.

    Cada concurso produz:

        25 dezenas × 26 features

    A construção é feita cronologicamente em uma única passagem.

    IMPORTANTE:
    As probabilidades condicionais são atualizadas usando somente
    transições já conhecidas naquele momento.

    Estado após concurso N
        ↓
    features
        ↓
    previsão do concurso N+1
    """

    NUM_DEZENAS = 25

    FEATURES = [
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
        "ranking_atraso",
    ]

    def __init__(self, df_bolas):
        self.df_bolas = df_bolas.copy()

        self.total_sorteios = len(
            self.df_bolas
        )

        print("Criando matriz binária V2...")

        self.matriz_binaria = (
            self._criar_matriz_binaria()
        )

        print("Criando somas cumulativas...")

        self.cumsum = (
            self._criar_cumsum()
        )

        self._cache_features = {}

        print(
            "Pré-calculando features V2..."
        )

        self._precalcular_features()

        print(
            f"-> {len(self._cache_features)} "
            f"concursos armazenados em memória."
        )

    # ========================================================
    # MATRIZ BINÁRIA
    # ========================================================

    def _criar_matriz_binaria(self):

        matriz = np.zeros(
            (
                self.total_sorteios,
                self.NUM_DEZENAS
            ),
            dtype=np.int8
        )

        for indice, sorteio in enumerate(
            self.df_bolas.to_numpy()
        ):

            for bola in sorteio:

                bola = int(bola)

                if not 1 <= bola <= 25:

                    raise ValueError(
                        f"Dezena inválida: {bola}"
                    )

                matriz[
                    indice,
                    bola - 1
                ] = 1

        return matriz

    # ========================================================
    # SOMA CUMULATIVA
    # ========================================================

    def _criar_cumsum(self):

        zero = np.zeros(
            (
                1,
                self.NUM_DEZENAS
            ),
            dtype=np.int32
        )

        acumulado = np.cumsum(
            self.matriz_binaria,
            axis=0,
            dtype=np.int32
        )

        return np.vstack(
            [
                zero,
                acumulado
            ]
        )

    # ========================================================
    # FREQUÊNCIAS
    # ========================================================

    def _frequencia_janela(
        self,
        indice,
        janela
    ):

        inicio = max(
            0,
            indice - janela + 1
        )

        fim = indice + 1

        tamanho = (
            fim - inicio
        )

        contagem = (
            self.cumsum[fim]
            - self.cumsum[inicio]
        )

        return (
            contagem.astype(
                np.float32
            )
            / tamanho
        )

    # ========================================================
    # MEDIANA DO HISTOGRAMA
    # ========================================================

    @staticmethod
    def _mediana_counter(
        contador,
        total
    ):

        if total == 0:
            return 0.0

        alvo_1 = (
            (total - 1) // 2
        )

        alvo_2 = (
            total // 2
        )

        acumulado = 0

        valor_1 = None
        valor_2 = None

        for valor in sorted(
            contador
        ):

            acumulado += (
                contador[valor]
            )

            if (
                valor_1 is None
                and acumulado > alvo_1
            ):
                valor_1 = valor

            if (
                valor_2 is None
                and acumulado > alvo_2
            ):
                valor_2 = valor
                break

        return (
            valor_1 + valor_2
        ) / 2.0

    # ========================================================
    # PERCENTIL DO ATRASO
    # ========================================================

    @staticmethod
    def _percentil_counter(
        contador,
        total,
        atraso
    ):

        if total == 0:
            return 0.0

        quantidade = sum(
            qtd
            for valor, qtd
            in contador.items()
            if valor <= atraso
        )

        return (
            quantidade / total
        )

    # ========================================================
    # RANKING
    # ========================================================

    @staticmethod
    def _ranking_desc(
        valores
    ):

        return (
            np.argsort(
                np.argsort(
                    -valores
                )
            )
            + 1
        )

    # ========================================================
    # PRÉ-CÁLCULO
    # ========================================================

    def _precalcular_features(self):

        # ----------------------------------------------------
        # Estado atual de cada dezena
        # ----------------------------------------------------

        atrasos = np.zeros(
            self.NUM_DEZENAS,
            dtype=np.int32
        )

        sequencias = np.zeros(
            self.NUM_DEZENAS,
            dtype=np.int32
        )

        ultima_ocorrencia = np.full(
            self.NUM_DEZENAS,
            -1,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Estatísticas dos intervalos de ausência
        # ----------------------------------------------------

        qtd_intervalos = np.zeros(
            self.NUM_DEZENAS,
            dtype=np.int32
        )

        soma_intervalos = np.zeros(
            self.NUM_DEZENAS,
            dtype=np.float64
        )

        soma_quadrados = np.zeros(
            self.NUM_DEZENAS,
            dtype=np.float64
        )

        max_intervalo = np.zeros(
            self.NUM_DEZENAS,
            dtype=np.int32
        )

        hist_intervalos = [
            Counter()
            for _ in range(
                self.NUM_DEZENAS
            )
        ]

        # ----------------------------------------------------
        # P(sair | atraso)
        # ----------------------------------------------------

        atraso_amostras = [
            defaultdict(int)
            for _ in range(
                self.NUM_DEZENAS
            )
        ]

        atraso_sucessos = [
            defaultdict(int)
            for _ in range(
                self.NUM_DEZENAS
            )
        ]

        # ----------------------------------------------------
        # P(repetir | sequência)
        # ----------------------------------------------------

        repeticao_amostras = [
            defaultdict(int)
            for _ in range(
                self.NUM_DEZENAS
            )
        ]

        repeticao_sucessos = [
            defaultdict(int)
            for _ in range(
                self.NUM_DEZENAS
            )
        ]

        # ====================================================
        # LOOP CRONOLÓGICO ÚNICO
        # ====================================================

        for indice in range(
            self.total_sorteios
        ):

            resultado = (
                self.matriz_binaria[
                    indice
                ]
            )

            # =================================================
            # 1. REGISTRAR A TRANSIÇÃO
            #
            # Estado conhecido em N-1
            # +
            # resultado de N.
            #
            # Isso reproduz o cálculo histórico:
            #
            # P(resultado seguinte | estado anterior)
            # =================================================

            if indice > 0:

                for d in range(
                    self.NUM_DEZENAS
                ):

                    saiu_agora = int(
                        resultado[d]
                    )

                    atraso_anterior = int(
                        atrasos[d]
                    )

                    sequencia_anterior = int(
                        sequencias[d]
                    )

                    # -----------------------------------------
                    # Atraso
                    # -----------------------------------------

                    if atraso_anterior > 0:

                        atraso_amostras[d][
                            atraso_anterior
                        ] += 1

                        if saiu_agora:

                            atraso_sucessos[d][
                                atraso_anterior
                            ] += 1

                    # -----------------------------------------
                    # Sequência
                    # -----------------------------------------

                    if sequencia_anterior > 0:

                        repeticao_amostras[d][
                            sequencia_anterior
                        ] += 1

                        if saiu_agora:

                            repeticao_sucessos[d][
                                sequencia_anterior
                            ] += 1

            # =================================================
            # 2. ATUALIZAR ESTADO COM O RESULTADO ATUAL
            # =================================================

            for d in range(
                self.NUM_DEZENAS
            ):

                if resultado[d] == 1:

                    # -----------------------------------------
                    # Fecha um intervalo de ausência
                    # -----------------------------------------

                    if ultima_ocorrencia[d] >= 0:

                        intervalo = (
                            indice
                            - ultima_ocorrencia[d]
                            - 1
                        )

                        qtd_intervalos[d] += 1

                        soma_intervalos[d] += (
                            intervalo
                        )

                        soma_quadrados[d] += (
                            intervalo
                            * intervalo
                        )

                        hist_intervalos[d][
                            intervalo
                        ] += 1

                        max_intervalo[d] = max(
                            max_intervalo[d],
                            intervalo
                        )

                    ultima_ocorrencia[d] = (
                        indice
                    )

                    atrasos[d] = 0

                    sequencias[d] += 1

                else:

                    atrasos[d] += 1

                    sequencias[d] = 0

            # =================================================
            # 3. FREQUÊNCIAS
            # =================================================

            freq_5 = (
                self._frequencia_janela(
                    indice,
                    5
                )
            )

            freq_10 = (
                self._frequencia_janela(
                    indice,
                    10
                )
            )

            freq_20 = (
                self._frequencia_janela(
                    indice,
                    20
                )
            )

            freq_50 = (
                self._frequencia_janela(
                    indice,
                    50
                )
            )

            freq_100 = (
                self._frequencia_janela(
                    indice,
                    100
                )
            )

            freq_200 = (
                self._frequencia_janela(
                    indice,
                    200
                )
            )

            freq_historica = (
                self.cumsum[
                    indice + 1
                ].astype(
                    np.float32
                )
                / (indice + 1)
            )

            # =================================================
            # 4. ESTATÍSTICAS RELATIVAS
            # =================================================

            media_freq20 = float(
                np.mean(
                    freq_20
                )
            )

            media_atrasos = float(
                np.mean(
                    atrasos
                )
            )

            ranking_freq = (
                self._ranking_desc(
                    freq_20
                )
            )

            ranking_atraso = (
                self._ranking_desc(
                    atrasos
                )
            )

            features_concurso = np.zeros(
                (
                    self.NUM_DEZENAS,
                    self.quantidade_features()
                ),
                dtype=np.float32
            )

            # =================================================
            # 5. FEATURES DAS 25 DEZENAS
            # =================================================

            for d in range(
                self.NUM_DEZENAS
            ):

                total_intervalos = int(
                    qtd_intervalos[d]
                )

                # ---------------------------------------------
                # Estatísticas de intervalos
                # ---------------------------------------------

                if total_intervalos > 0:

                    media_intervalo = (
                        soma_intervalos[d]
                        / total_intervalos
                    )

                    variancia = (
                        (
                            soma_quadrados[d]
                            / total_intervalos
                        )
                        - (
                            media_intervalo
                            ** 2
                        )
                    )

                    variancia = max(
                        0.0,
                        variancia
                    )

                    desvio_intervalo = (
                        np.sqrt(
                            variancia
                        )
                    )

                    mediana_intervalo = (
                        self._mediana_counter(
                            hist_intervalos[d],
                            total_intervalos
                        )
                    )

                    percentil_atraso = (
                        self._percentil_counter(
                            hist_intervalos[d],
                            total_intervalos,
                            int(
                                atrasos[d]
                            )
                        )
                    )

                else:

                    media_intervalo = 0.0

                    mediana_intervalo = 0.0

                    desvio_intervalo = 0.0

                    percentil_atraso = 0.0

                # ---------------------------------------------
                # P(sair | atraso atual)
                # ---------------------------------------------

                atraso_atual = int(
                    atrasos[d]
                )

                if atraso_atual > 0:

                    amostras_atraso = (
                        atraso_amostras[d].get(
                            atraso_atual,
                            0
                        )
                    )

                    sucessos_atraso = (
                        atraso_sucessos[d].get(
                            atraso_atual,
                            0
                        )
                    )

                else:

                    amostras_atraso = 0

                    sucessos_atraso = 0

                if amostras_atraso > 0:

                    prob_sair_atraso = (
                        sucessos_atraso
                        / amostras_atraso
                    )

                    lift_atraso = (
                        prob_sair_atraso
                        - 0.60
                    )

                else:

                    prob_sair_atraso = 0.0

                    lift_atraso = 0.0

                # ---------------------------------------------
                # P(repetir | sequência atual)
                # ---------------------------------------------

                seq_atual = int(
                    sequencias[d]
                )

                if seq_atual > 0:

                    amostras_repeticao = (
                        repeticao_amostras[d].get(
                            seq_atual,
                            0
                        )
                    )

                    sucessos_repeticao = (
                        repeticao_sucessos[d].get(
                            seq_atual,
                            0
                        )
                    )

                else:

                    amostras_repeticao = 0

                    sucessos_repeticao = 0

                if amostras_repeticao > 0:

                    prob_repetir = (
                        sucessos_repeticao
                        / amostras_repeticao
                    )

                    lift_repeticao = (
                        prob_repetir
                        - 0.60
                    )

                else:

                    prob_repetir = 0.0

                    lift_repeticao = 0.0

                # ---------------------------------------------
                # Vetor final
                # ---------------------------------------------

                features_concurso[d] = [
                    (d + 1) / 25.0,

                    freq_5[d],
                    freq_10[d],
                    freq_20[d],
                    freq_50[d],
                    freq_100[d],
                    freq_200[d],
                    freq_historica[d],

                    atraso_atual,
                    percentil_atraso,
                    media_intervalo,
                    mediana_intervalo,
                    desvio_intervalo,
                    max_intervalo[d],

                    seq_atual,

                    prob_sair_atraso,
                    np.log1p(
                        amostras_atraso
                    ),

                    prob_repetir,
                    np.log1p(
                        amostras_repeticao
                    ),

                    lift_atraso,
                    lift_repeticao,

                    resultado[d],

                    (
                        freq_20[d]
                        - media_freq20
                    ),

                    (
                        atraso_atual
                        - media_atrasos
                    ),

                    ranking_freq[d]
                    / 25.0,

                    ranking_atraso[d]
                    / 25.0,
                ]

            self._cache_features[
                indice
            ] = features_concurso

            if (
                indice > 0
                and indice % 500 == 0
            ):

                print(
                    f"  {indice}/"
                    f"{self.total_sorteios}"
                )

    # ========================================================
    # INTERFACE NORMAL
    # ========================================================

    def calcular_features_concurso(
        self,
        indice
    ):

        return (
            self._cache_features[
                indice
            ].copy()
        )

    # ========================================================
    # DATASET
    # ========================================================

    def construir_dataset(
        self,
        janela_minima=200
    ):

        quantidade_concursos = (
            self.total_sorteios
            - janela_minima
            - 1
        )

        quantidade_linhas = (
            quantidade_concursos
            * self.NUM_DEZENAS
        )

        print()
        print(
            "Montando dataset V2..."
        )

        print(
            f"-> Concursos úteis: "
            f"{quantidade_concursos}"
        )

        print(
            f"-> Linhas: "
            f"{quantidade_linhas:,}"
        )

        X = np.empty(
            (
                quantidade_linhas,
                self.quantidade_features()
            ),
            dtype=np.float32
        )

        y = np.empty(
            quantidade_linhas,
            dtype=np.int8
        )

        indices_target = np.empty(
            quantidade_linhas,
            dtype=np.int32
        )

        dezenas = np.empty(
            quantidade_linhas,
            dtype=np.int8
        )

        posicao = 0

        for indice in range(
            janela_minima,
            self.total_sorteios - 1
        ):

            fim = (
                posicao
                + 25
            )

            X[
                posicao:fim
            ] = (
                self._cache_features[
                    indice
                ]
            )

            y[
                posicao:fim
            ] = (
                self.matriz_binaria[
                    indice + 1
                ]
            )

            indices_target[
                posicao:fim
            ] = (
                indice + 1
            )

            dezenas[
                posicao:fim
            ] = np.arange(
                1,
                26
            )

            posicao = fim

        return (
            X,
            y,
            indices_target,
            dezenas
        )

    # ========================================================
    # IMPLEMENTAÇÃO LENTA PARA VALIDAÇÃO
    # ========================================================

    def _features_lentas(
        self,
        indice
    ):
        """
        Recalcula as features pelo método direto.

        NÃO é utilizado no treinamento.

        Existe apenas para validar matematicamente
        a implementação rápida.
        """

        passado = (
            self.matriz_binaria[
                :indice + 1
            ]
        )

        estatisticas = []

        for d in range(25):

            serie = (
                passado[:, d]
            )

            def freq(janela):

                tamanho = min(
                    janela,
                    len(serie)
                )

                return float(
                    np.mean(
                        serie[
                            -tamanho:
                        ]
                    )
                )

            # -----------------------------------------------
            # atraso
            # -----------------------------------------------

            atraso = 0

            for valor in reversed(
                serie
            ):

                if valor == 1:
                    break

                atraso += 1

            # -----------------------------------------------
            # sequência
            # -----------------------------------------------

            sequencia = 0

            for valor in reversed(
                serie
            ):

                if valor == 0:
                    break

                sequencia += 1

            # -----------------------------------------------
            # intervalos
            # -----------------------------------------------

            posicoes = np.where(
                serie == 1
            )[0]

            if len(posicoes) >= 2:

                intervalos = (
                    np.diff(
                        posicoes
                    )
                    - 1
                )

                media_intervalo = float(
                    np.mean(
                        intervalos
                    )
                )

                mediana_intervalo = float(
                    np.median(
                        intervalos
                    )
                )

                desvio_intervalo = float(
                    np.std(
                        intervalos
                    )
                )

                max_intervalo = int(
                    np.max(
                        intervalos
                    )
                )

                percentil = float(
                    np.mean(
                        intervalos <= atraso
                    )
                )

            else:

                intervalos = []

                media_intervalo = 0.0
                mediana_intervalo = 0.0
                desvio_intervalo = 0.0
                max_intervalo = 0
                percentil = 0.0

            # -----------------------------------------------
            # P(sair | atraso)
            # -----------------------------------------------

            prob_atraso = 0.0
            amostras_atraso = 0

            if atraso > 0:

                estado = 0
                sucessos = 0

                for i in range(
                    len(serie) - 1
                ):

                    if serie[i] == 1:
                        estado = 0
                    else:
                        estado += 1

                    if estado == atraso:

                        amostras_atraso += 1

                        if serie[i + 1] == 1:
                            sucessos += 1

                if amostras_atraso > 0:

                    prob_atraso = (
                        sucessos
                        / amostras_atraso
                    )

            # -----------------------------------------------
            # P(repetir | sequência)
            # -----------------------------------------------

            prob_repetir = 0.0
            amostras_repeticao = 0

            if sequencia > 0:

                estado = 0
                sucessos = 0

                for i in range(
                    len(serie) - 1
                ):

                    if serie[i] == 1:
                        estado += 1
                    else:
                        estado = 0

                    if estado == sequencia:

                        amostras_repeticao += 1

                        if serie[i + 1] == 1:
                            sucessos += 1

                if (
                    amostras_repeticao
                    > 0
                ):

                    prob_repetir = (
                        sucessos
                        / amostras_repeticao
                    )

            estatisticas.append({
                "dezena": d + 1,

                "freq_5": freq(5),
                "freq_10": freq(10),
                "freq_20": freq(20),
                "freq_50": freq(50),
                "freq_100": freq(100),
                "freq_200": freq(200),

                "freq_historica":
                    float(
                        np.mean(
                            serie
                        )
                    ),

                "atraso":
                    atraso,

                "percentil":
                    percentil,

                "media":
                    media_intervalo,

                "mediana":
                    mediana_intervalo,

                "desvio":
                    desvio_intervalo,

                "max":
                    max_intervalo,

                "sequencia":
                    sequencia,

                "prob_atraso":
                    prob_atraso,

                "amostras_atraso":
                    amostras_atraso,

                "prob_repetir":
                    prob_repetir,

                "amostras_repeticao":
                    amostras_repeticao,

                "saiu_anterior":
                    int(
                        serie[-1]
                    ),
            })

        freq20 = np.asarray(
            [
                x["freq_20"]
                for x in estatisticas
            ]
        )

        atrasos = np.asarray(
            [
                x["atraso"]
                for x in estatisticas
            ]
        )

        media_freq = np.mean(
            freq20
        )

        media_atrasos = np.mean(
            atrasos
        )

        ranking_freq = (
            self._ranking_desc(
                freq20
            )
        )

        ranking_atraso = (
            self._ranking_desc(
                atrasos
            )
        )

        resultado = []

        for d, x in enumerate(
            estatisticas
        ):

            lift_atraso = (
                x["prob_atraso"] - 0.60
                if x["amostras_atraso"] > 0
                else 0.0
            )

            lift_repeticao = (
                x["prob_repetir"] - 0.60
                if x["amostras_repeticao"] > 0
                else 0.0
            )

            resultado.append([
                x["dezena"] / 25.0,

                x["freq_5"],
                x["freq_10"],
                x["freq_20"],
                x["freq_50"],
                x["freq_100"],
                x["freq_200"],
                x["freq_historica"],

                x["atraso"],
                x["percentil"],
                x["media"],
                x["mediana"],
                x["desvio"],
                x["max"],

                x["sequencia"],

                x["prob_atraso"],
                np.log1p(
                    x["amostras_atraso"]
                ),

                x["prob_repetir"],
                np.log1p(
                    x["amostras_repeticao"]
                ),

                lift_atraso,
                lift_repeticao,

                x["saiu_anterior"],

                x["freq_20"]
                - media_freq,

                x["atraso"]
                - media_atrasos,

                ranking_freq[d]
                / 25.0,

                ranking_atraso[d]
                / 25.0,
            ])

        return np.asarray(
            resultado,
            dtype=np.float32
        )

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    def validar_equivalencia(
        self
    ):
        """
        Compara FAST x método lento.

        Se uma feature estiver diferente,
        interrompe imediatamente.
        """

        indices = [
            200,
            500,
            1000,
            2000,
            3000,
            self.total_sorteios - 2
        ]

        print()
        print("=" * 70)
        print(
            "VALIDANDO FEATURES V2"
        )
        print("=" * 70)

        for indice in indices:

            if indice >= (
                self.total_sorteios
            ):
                continue

            rapido = (
                self.calcular_features_concurso(
                    indice
                )
            )

            lento = (
                self._features_lentas(
                    indice
                )
            )

            correto = np.allclose(
                rapido,
                lento,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True
            )

            if not correto:

                diferencas = np.abs(
                    rapido - lento
                )

                linha, coluna = (
                    np.unravel_index(
                        np.argmax(
                            diferencas
                        ),
                        diferencas.shape
                    )
                )

                nome_feature = (
                    self.FEATURES[
                        coluna
                    ]
                )

                raise AssertionError(
                    "\nERRO DE EQUIVALÊNCIA!\n"
                    f"Concurso índice: {indice}\n"
                    f"Dezena: {linha + 1}\n"
                    f"Feature: {nome_feature}\n"
                    f"Rápido: {rapido[linha, coluna]}\n"
                    f"Lento: {lento[linha, coluna]}\n"
                    f"Diferença: "
                    f"{diferencas[linha, coluna]}"
                )

            print(
                f"Índice {indice}: OK"
            )

        print()
        print(
            "VALIDAÇÃO CONCLUÍDA:"
        )

        print(
            "implementação rápida == "
            "implementação de referência."
        )

    # ========================================================
    # PRÓXIMO CONCURSO
    # ========================================================

    def features_proximo_concurso(
        self
    ):

        return (
            self.calcular_features_concurso(
                self.total_sorteios - 1
            )
        )

    @classmethod
    def nomes_features(cls):

        return list(
            cls.FEATURES
        )

    @classmethod
    def quantidade_features(cls):

        return len(
            cls.FEATURES
        )