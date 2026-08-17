from fechamento import (
    gerar_todas_combinacoes,
    selecionar_por_diversidade
)


class GeradorJogos:

    def __init__(
        self,
        filtros=None
    ):
        self.filtros = (
            filtros or []
        )

    def aplicar_filtros(
        self,
        jogos,
        stats=None
    ):
        """
        Aplica os filtros configurados.

        Se nenhum filtro for informado,
        retorna todos os jogos.
        """

        if not self.filtros:
            return list(jogos)

        jogos_validos = []

        for jogo in jogos:

            valido = all(
                filtro.eh_valido(
                    jogo,
                    stats
                )
                for filtro
                in self.filtros
            )

            if valido:
                jogos_validos.append(
                    jogo
                )

        return jogos_validos

    def gerar(
        self,
        dezenas_candidatas,
        quantidade_jogos=50,
        stats=None
    ):
        """
        Pipeline completo:

            dezenas candidatas
                ↓
            todas combinações
                ↓
            filtros
                ↓
            seleção por diversidade
        """

        todas = (
            gerar_todas_combinacoes(
                dezenas_candidatas
            )
        )

        filtradas = (
            self.aplicar_filtros(
                todas,
                stats
            )
        )

        if not filtradas:
            raise ValueError(
                "Todos os jogos foram removidos "
                "pelos filtros."
            )

        quantidade_real = min(
            quantidade_jogos,
            len(filtradas)
        )

        selecionadas = (
            selecionar_por_diversidade(
                filtradas,
                quantidade_real
            )
        )

        return {
            "total_combinacoes":
                len(todas),

            "total_apos_filtros":
                len(filtradas),

            "jogos":
                selecionadas
        }


def imprimir_jogos(
    jogos
):
    """
    Exibe os jogos formatados.
    """

    print()

    print("=" * 60)
    print("JOGOS GERADOS")
    print("=" * 60)

    for numero, jogo in enumerate(
        jogos,
        start=1
    ):

        dezenas = " ".join(
            f"{dezena:02d}"
            for dezena
            in jogo
        )

        print(
            f"Jogo {numero:03d}: "
            f"{dezenas}"
        )


def calcular_custo(
    quantidade_jogos,
    custo_unitario=3.50
):
    return (
        quantidade_jogos
        * custo_unitario
    )