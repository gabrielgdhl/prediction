from itertools import combinations


def gerar_todas_combinacoes(
    dezenas_candidatas,
    tamanho_jogo=15
):
    """
    Gera todas as combinações possíveis.

    Exemplo:

        18 dezenas candidatas
            ↓
        C(18, 15)
            ↓
        816 jogos
    """

    dezenas = sorted(
        set(dezenas_candidatas)
    )

    if len(dezenas) < tamanho_jogo:
        raise ValueError(
            f"São necessárias pelo menos "
            f"{tamanho_jogo} dezenas."
        )

    return list(
        combinations(
            dezenas,
            tamanho_jogo
        )
    )


def jogo_para_mask(jogo):
    """
    Converte um jogo para bitmask.

    Isso torna comparação entre jogos
    muito mais rápida.
    """

    mask = 0

    for dezena in jogo:
        mask |= (
            1 << (dezena - 1)
        )

    return mask


def distancia_entre_jogos(
    mask_a,
    mask_b,
    tamanho_jogo=15
):
    """
    Mede quantas dezenas diferem entre dois jogos.

    Para dois jogos de 15 dezenas:

        distância 0
            = jogos idênticos

        distância maior
            = jogos mais diferentes
    """

    intersecao = (
        mask_a
        & mask_b
    ).bit_count()

    return (
        tamanho_jogo
        - intersecao
    )


def selecionar_por_diversidade(
    jogos,
    quantidade
):
    """
    Seleção gulosa Max-Min.

    Escolhe jogos tentando maximizar
    a diversidade entre os selecionados.
    """

    if quantidade <= 0:
        return []

    if quantidade >= len(jogos):
        return list(jogos)

    masks = [
        jogo_para_mask(jogo)
        for jogo in jogos
    ]

    selecionados_indices = [
        0
    ]

    restantes = set(
        range(
            1,
            len(jogos)
        )
    )

    while (
        len(selecionados_indices)
        < quantidade
    ):

        melhor_indice = None
        melhor_score = -1

        masks_selecionados = [
            masks[i]
            for i
            in selecionados_indices
        ]

        for indice in restantes:

            mask_atual = masks[
                indice
            ]

            menor_distancia = min(
                distancia_entre_jogos(
                    mask_atual,
                    mask_selecionado
                )
                for mask_selecionado
                in masks_selecionados
            )

            if (
                menor_distancia
                > melhor_score
            ):

                melhor_score = (
                    menor_distancia
                )

                melhor_indice = (
                    indice
                )

        selecionados_indices.append(
            melhor_indice
        )

        restantes.remove(
            melhor_indice
        )

    return [
        jogos[indice]
        for indice
        in selecionados_indices
    ]