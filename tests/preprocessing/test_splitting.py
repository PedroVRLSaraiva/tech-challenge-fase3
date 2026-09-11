import numpy as np
import pandas as pd

from src.preprocessing.splitting import dividir_treino_validacao_teste


def _dados_sinteticos(n_2023=2000, n_2024=500, taxa_risco=0.4, seed=0):
    rng = np.random.default_rng(seed)
    ano = [2023] * n_2023 + [2024] * n_2024
    em_risco = rng.binomial(1, taxa_risco, size=n_2023 + n_2024)
    return pd.DataFrame({"ano": ano, "em_risco": em_risco, "id": range(n_2023 + n_2024)})


def test_proporcoes_do_split_dentro_de_2023():
    df = _dados_sinteticos()
    partes = dividir_treino_validacao_teste(df)

    total_2023 = len(df[df["ano"] == 2023])
    assert abs(len(partes["treino"]) / total_2023 - 0.70) < 0.02
    assert abs(len(partes["validacao"]) / total_2023 - 0.15) < 0.02
    assert abs(len(partes["teste_mesmo_ano"]) / total_2023 - 0.15) < 0.02


def test_teste_futuro_e_2024_inteiro_e_nao_e_dividido():
    df = _dados_sinteticos()
    partes = dividir_treino_validacao_teste(df)

    assert (partes["teste_futuro"]["ano"] == 2024).all()
    assert len(partes["teste_futuro"]) == len(df[df["ano"] == 2024])


def test_estratificacao_preserva_balanceamento_da_classe():
    df = _dados_sinteticos(taxa_risco=0.4)
    partes = dividir_treino_validacao_teste(df)
    taxa_original = df[df["ano"] == 2023]["em_risco"].mean()

    for nome in ["treino", "validacao", "teste_mesmo_ano"]:
        taxa_parte = partes[nome]["em_risco"].mean()
        assert abs(taxa_parte - taxa_original) < 0.03, nome


def test_nenhuma_linha_de_2023_aparece_em_mais_de_uma_parte():
    df = _dados_sinteticos()
    partes = dividir_treino_validacao_teste(df)

    ids_treino = set(partes["treino"]["id"])
    ids_validacao = set(partes["validacao"]["id"])
    ids_teste = set(partes["teste_mesmo_ano"]["id"])

    assert ids_treino.isdisjoint(ids_validacao)
    assert ids_treino.isdisjoint(ids_teste)
    assert ids_validacao.isdisjoint(ids_teste)
    assert len(ids_treino | ids_validacao | ids_teste) == len(df[df["ano"] == 2023])
