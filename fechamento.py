from itertools import combinations

def selecionar_melhores_jogos(dezenas_restantes, dezenas_imunes, probabilidades_ml, filtros_pipeline, stats, quantidade_desejada=100):
    todas_combinacoes = list(combinations(dezenas_restantes, 15))
    jogos_pontuados = []
    
    for jogo in todas_combinacoes:
        if not all(filtro.eh_valido(jogo, stats) for filtro in filtros_pipeline):
            continue
            
        score = 0.0
        jogo_set = set(jogo)
        
        # Critério A: Soma das probabilidades
        score_ml = 0
        for num in jogo:
            if len(probabilidades_ml[num - 1][0]) == 2:
                score_ml += probabilidades_ml[num - 1][0][1]
        score += score_ml
        
        # Critério B: Peso alto para as dezenas imunes (garante a presença delas)
        qtd_imunes = len(jogo_set.intersection(set(dezenas_imunes)))
        score += (qtd_imunes * 5.0)
        
        jogos_pontuados.append((score, jogo))
        
    jogos_pontuados.sort(key=lambda x: x[0], reverse=True)
    melhores_jogos_selecionados = jogos_pontuados[:quantidade_desejada]
    
    return [jogo for score, jogo in melhores_jogos_selecionados]

def relatorio_financeiro_parametrizado(jogos_finais):
    custo_volante = 3.50
    total_jogos = len(jogos_finais)
    investimento = total_jogos * custo_volante
    
    print("\n--- Resumo Financeiro ---")
    print(f"Quantidade de apostas geradas: {total_jogos}")
    print(f"Custo unitário por jogo: R$ {custo_volante:.2f}")
    print(f"Investimento total necessário: R$ {investimento:.2f}")