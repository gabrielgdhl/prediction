import numpy as np


class GeradorEstatisticasAvancadas:
    """
    Responsável por transformar o histórico da Lotofácil
    em features estatísticas para Machine Learning.

    Regra temporal adotada:

        calcular_features_no_indice(i)

    usa somente os concursos:

        0 ... i

    para tentar prever:

        i + 1

    Dessa forma evitamos data leakage.
    """

    FEATURES_POR_DEZENA = [
        # Frequências absolutas
        "freq_5",
        "freq_10",
        "freq_20",
        "freq_50",
        "freq_100",
        "freq_200",

        # Frequências normalizadas
        "taxa_5",
        "taxa_10",
        "taxa_20",
        "taxa_50",
        "taxa_100",
        "taxa_200",

        # Histórico
        "freq_historica_normalizada",

        # Atrasos
        "atraso_atual",
        "max_atraso_historico",
        "media_intervalo",
        "mediana_intervalo",
        "desvio_intervalo",
        "z_score_atraso",

        # Concurso anterior
        "apareceu_ultimo",

        # Tendências
        "tendencia_5_20",
        "tendencia_10_50",
        "tendencia_20_100",

        # Relação atraso atual x histórico
        "distancia_max_atraso",
        "ratio_atraso_max",
    ]

    def __init__(self, df_bolas):
        self.df_bolas = df_bolas.copy()

        self.total_sorteios = len(
            self.df_bolas
        )

        self.matriz_binaria = (
            self._criar_matriz_binaria()
        )

    # =========================================================
    # MATRIZ BINÁRIA
    # =========================================================

    def _criar_matriz_binaria(self):
        """
        Converte:

            Bola1 ... Bola15

        em uma matriz binária:

            concurso x 25 dezenas

        Exemplo:

            índice 0 = dezena 01
            índice 1 = dezena 02
            ...
            índice 24 = dezena 25

        Valor:

            1 = saiu
            0 = não saiu
        """

        matriz = np.zeros(
            (
                self.total_sorteios,
                25
            ),
            dtype=np.int8
        )

        for indice_sorteio, sorteio in enumerate(
            self.df_bolas.to_numpy()
        ):

            for bola in sorteio:
                bola = int(bola)

                if bola < 1 or bola > 25:
                    raise ValueError(
                        f"Dezena inválida encontrada: {bola}"
                    )

                matriz[
                    indice_sorteio,
                    bola - 1
                ] = 1

        return matriz

    # =========================================================
    # UTILITÁRIOS
    # =========================================================

    @staticmethod
    def _taxa(
        quantidade,
        tamanho_janela
    ):
        if tamanho_janela <= 0:
            return 0.0

        return float(
            quantidade / tamanho_janela
        )

    @staticmethod
    def _frequencia_janela(
        serie,
        janela
    ):
        tamanho = min(
            janela,
            len(serie)
        )

        if tamanho == 0:
            return 0, 0

        frequencia = int(
            np.sum(
                serie[-tamanho:]
            )
        )

        return (
            frequencia,
            tamanho
        )

    # =========================================================
    # FEATURES
    # =========================================================

    def calcular_features_no_indice(
        self,
        indice
    ):
        """
        Calcula as features existentes APÓS o concurso `indice`.

        Exemplo:

            calcular_features_no_indice(1000)

        utiliza apenas os concursos:

            0 ... 1000

        para posteriormente prever:

            concurso 1001
        """

        if indice < 0:
            raise ValueError(
                "O índice não pode ser negativo."
            )

        if indice >= self.total_sorteios:
            raise ValueError(
                f"Índice {indice} fora da base. "
                f"Total: {self.total_sorteios}"
            )

        matriz_passado = (
            self.matriz_binaria[
                :indice + 1
            ]
        )

        tamanho_passado = len(
            matriz_passado
        )

        # O último resultado conhecido naquele momento.
        ultimo_sorteio = (
            self.matriz_binaria[
                indice
            ]
        )

        features_totais = []

        # =====================================================
        # PROCESSA CADA UMA DAS 25 DEZENAS
        # =====================================================

        for indice_dezena in range(25):

            serie = (
                matriz_passado[
                    :,
                    indice_dezena
                ]
            )

            # =================================================
            # FREQUÊNCIAS
            # =================================================

            freq_5, tamanho_5 = (
                self._frequencia_janela(
                    serie,
                    5
                )
            )

            freq_10, tamanho_10 = (
                self._frequencia_janela(
                    serie,
                    10
                )
            )

            freq_20, tamanho_20 = (
                self._frequencia_janela(
                    serie,
                    20
                )
            )

            freq_50, tamanho_50 = (
                self._frequencia_janela(
                    serie,
                    50
                )
            )

            freq_100, tamanho_100 = (
                self._frequencia_janela(
                    serie,
                    100
                )
            )

            freq_200, tamanho_200 = (
                self._frequencia_janela(
                    serie,
                    200
                )
            )

            # =================================================
            # TAXAS DE OCORRÊNCIA
            # =================================================

            taxa_5 = self._taxa(
                freq_5,
                tamanho_5
            )

            taxa_10 = self._taxa(
                freq_10,
                tamanho_10
            )

            taxa_20 = self._taxa(
                freq_20,
                tamanho_20
            )

            taxa_50 = self._taxa(
                freq_50,
                tamanho_50
            )

            taxa_100 = self._taxa(
                freq_100,
                tamanho_100
            )

            taxa_200 = self._taxa(
                freq_200,
                tamanho_200
            )

            # =================================================
            # FREQUÊNCIA HISTÓRICA
            # =================================================

            freq_historica = int(
                np.sum(serie)
            )

            freq_historica_normalizada = (
                freq_historica
                / tamanho_passado
            )

            # =================================================
            # ATRASO ATUAL
            # =================================================

            atraso_atual = 0

            for posicao in range(
                tamanho_passado - 1,
                -1,
                -1
            ):

                if serie[posicao] == 1:
                    break

                atraso_atual += 1

            # =================================================
            # INTERVALOS ENTRE OCORRÊNCIAS
            # =================================================

            ocorrencias = np.flatnonzero(
                serie == 1
            )

            if len(ocorrencias) >= 2:

                # Exemplo:
                #
                # saiu no índice 10
                # saiu no índice 13
                #
                # ficou sem sair em:
                #
                # 11 e 12
                #
                # atraso = 2
                intervalos = (
                    np.diff(
                        ocorrencias
                    ) - 1
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

                # IMPORTANTE:
                #
                # O atraso atual NÃO entra aqui.
                #
                # Queremos descobrir qual era
                # o maior atraso HISTÓRICO
                # antes do estado atual.
                max_atraso_historico = int(
                    np.max(
                        intervalos
                    )
                )

            else:

                media_intervalo = 0.0
                mediana_intervalo = 0.0
                desvio_intervalo = 0.0
                max_atraso_historico = 0

            # =================================================
            # APARECEU NO ÚLTIMO CONCURSO?
            # =================================================

            apareceu_ultimo = int(
                ultimo_sorteio[
                    indice_dezena
                ]
            )

            # =================================================
            # Z-SCORE DO ATRASO
            # =================================================

            if desvio_intervalo > 0:

                z_score_atraso = (
                    atraso_atual
                    - media_intervalo
                ) / desvio_intervalo

            else:

                z_score_atraso = 0.0

            # =================================================
            # TENDÊNCIAS
            # =================================================

            # Positivo:
            # dezena está aparecendo mais no curto prazo.
            #
            # Negativo:
            # aparecendo menos recentemente.

            tendencia_5_20 = (
                taxa_5
                - taxa_20
            )

            tendencia_10_50 = (
                taxa_10
                - taxa_50
            )

            tendencia_20_100 = (
                taxa_20
                - taxa_100
            )

            # =================================================
            # ATRASO x MAIOR ATRASO
            # =================================================

            distancia_max_atraso = (
                max_atraso_historico
                - atraso_atual
            )

            if max_atraso_historico > 0:

                ratio_atraso_max = (
                    atraso_atual
                    / max_atraso_historico
                )

            else:

                ratio_atraso_max = 0.0

            # =================================================
            # FEATURES DA DEZENA
            # =================================================

            features_dezena = [
                freq_5,
                freq_10,
                freq_20,
                freq_50,
                freq_100,
                freq_200,

                taxa_5,
                taxa_10,
                taxa_20,
                taxa_50,
                taxa_100,
                taxa_200,

                freq_historica_normalizada,

                atraso_atual,
                max_atraso_historico,
                media_intervalo,
                mediana_intervalo,
                desvio_intervalo,
                z_score_atraso,

                apareceu_ultimo,

                tendencia_5_20,
                tendencia_10_50,
                tendencia_20_100,

                distancia_max_atraso,
                ratio_atraso_max,
            ]

            features_totais.extend(
                features_dezena
            )

        return np.asarray(
            features_totais,
            dtype=np.float32
        )

    # =========================================================
    # DATASET PARA MACHINE LEARNING
    # =========================================================

    def construir_dataset_ml(
        self,
        janela_minima=200
    ):
        """
        Monta:

            X = features conhecidas após concurso i

            y = resultado real do concurso i + 1

        Target:

            1 = saiu
            0 = não saiu

        Também retorna `indices`, contendo o índice
        do concurso que cada linha de y representa.
        """

        if janela_minima < 1:
            raise ValueError(
                "janela_minima deve ser maior que zero."
            )

        if janela_minima >= self.total_sorteios - 1:
            raise ValueError(
                "janela_minima é grande demais "
                "para o histórico disponível."
            )

        X = []
        y = []
        indices = []

        for indice in range(
            janela_minima,
            self.total_sorteios - 1
        ):

            # Features disponíveis após
            # o concurso atual.
            features = (
                self.calcular_features_no_indice(
                    indice
                )
            )

            # Resultado FUTURO.
            target = (
                self.matriz_binaria[
                    indice + 1
                ]
            )

            X.append(
                features
            )

            y.append(
                target
            )

            # Este índice representa
            # o concurso previsto.
            indices.append(
                indice + 1
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
                indices,
                dtype=np.int32
            )
        )

    # =========================================================
    # FEATURES PARA O PRÓXIMO CONCURSO
    # =========================================================

    def features_proximo_concurso(self):
        """
        Usa todo o histórico conhecido para gerar
        as features do próximo concurso ainda não realizado.
        """

        ultimo_indice = (
            self.total_sorteios - 1
        )

        return (
            self.calcular_features_no_indice(
                ultimo_indice
            )
        )

    # =========================================================
    # INFORMAÇÕES DA ESTRUTURA
    # =========================================================

    def quantidade_features_por_dezena(
        self
    ):
        return len(
            self.FEATURES_POR_DEZENA
        )

    def quantidade_features_total(
        self
    ):
        return (
            self.quantidade_features_por_dezena()
            * 25
        )

    def nomes_features(
        self
    ):
        """
        Retorna nomes como:

            dezena_01_freq_5
            dezena_01_freq_10
            ...
            dezena_25_ratio_atraso_max
        """

        nomes = []

        for dezena in range(
            1,
            26
        ):

            for feature in (
                self.FEATURES_POR_DEZENA
            ):

                nomes.append(
                    f"dezena_{dezena:02d}_{feature}"
                )

        return nomes