import pandas as pd
import numpy as np

# Importação dos módulos locais
from features import GeradorEstatisticas
from ml_core import treinar_maquina_preditiva, IdentificadorDezenasRuins
from filtros import MotorEstatisticas, FiltroSoma, FiltroImparesPares, FiltroPrimos, FiltroHistorico
from fechamento import selecionar_melhores_jogos, relatorio_financeiro_parametrizado

def executar_sistema():
    print("--- INICIANDO SISTEMA PREDITIVO ---")
    
    # 1. Carga de Dados (Base Real)
    print("Carregando base de dados histórica...")
    
    # Lê a planilha do Excel
    df_completo = pd.read_excel('lotofacil_resultados.xlsx')
    
    # TODO: Ajuste os nomes das colunas de acordo com o cabeçalho exato da sua planilha
    # Exemplo: se na planilha as colunas se chamam 'Bola1', 'Bola2', etc.
    colunas_das_bolas = [f'Bola{i}' for i in range(1, 16)]
    
    # Filtra o DataFrame para manter APENAS as 15 colunas de dezenas
    df_bolas = df_completo[colunas_das_bolas].copy()
    
    # Garante que os dados sejam números inteiros
    df_bolas = df_bolas.astype(int)
    
    print(f"Base carregada com sucesso! Total de concursos analisados: {len(df_bolas)}")
    
    # 2. Engenharia de Dados
    gerador = GeradorEstatisticas(df_bolas, janela_frequencia=15)
    X, y = gerador.construir_dataset()
    
    # 3. Treinamento ML
    modelo = treinar_maquina_preditiva(X, y)
    
    # 4. Seleção das Dezenas Ruins
    identificador = IdentificadorDezenasRuins(modelo, gerador)
    restantes, imunes, ruins, probs = identificador.selecionar_piores_sete()
    
    print(f"\n🔒 Imunes: {imunes}")
    print(f"❌ Excluídas: {ruins}")
    print(f"🎯 Restantes (Para combinação): {restantes}")
    
    # 5. Configuração dos Filtros Estruturais
    stats = MotorEstatisticas(df_bolas)
    filtros = [
        FiltroSoma(),
        FiltroImparesPares(),
        FiltroPrimos(),
        FiltroHistorico(df_bolas)
    ]
    
    # 6. Geração do Fechamento Parametrizado
    quantidade_alvo = 50  # Quantos jogos você quer registrar
    jogos_finais = selecionar_melhores_jogos(
        dezenas_restantes=restantes,
        dezenas_imunes=imunes,
        probabilidades_ml=probs,
        filtros_pipeline=filtros,
        stats=stats,
        quantidade_desejada=quantidade_alvo
    )
    
    # 7. Resultados
    print(f"\n--- TOP 3 JOGOS GERADOS ---")
    for i, jogo in enumerate(jogos_finais[:50]):
        print(f"Jogo {i+1}: {jogo}")
        
    relatorio_financeiro_parametrizado(jogos_finais)

if __name__ == "__main__":
    executar_sistema()