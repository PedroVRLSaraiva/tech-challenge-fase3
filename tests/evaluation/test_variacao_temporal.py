import numpy as np
import pandas as pd

from src.evaluation.variacao_temporal import (
    comparar_distribuicoes_categoricas,
    comparar_distribuicoes_numericas,
)


def test_comparar_distribuicoes_numericas_detecta_distribuicao_estavel():
    rng = np.random.default_rng(0)
    df_2023 = pd.DataFrame({"x": rng.normal(70, 5, 2000)})
    df_2024 = pd.DataFrame({"x": rng.normal(70, 5, 2000)})

    resultado = comparar_distribuicoes_numericas(df_2023, df_2024, ["x"])

    linha = resultado.set_index("coluna").loc["x"]
    assert linha["p_valor"] > 0.05
    assert linha["distribuicao_mudou"] is np.False_ or linha["distribuicao_mudou"] is False


def test_comparar_distribuicoes_numericas_detecta_mudanca_real():
    rng = np.random.default_rng(0)
    df_2023 = pd.DataFrame({"x": rng.normal(70, 5, 2000)})
    df_2024 = pd.DataFrame({"x": rng.normal(50, 5, 2000)})  # deslocamento grande

    resultado = comparar_distribuicoes_numericas(df_2023, df_2024, ["x"])

    linha = resultado.set_index("coluna").loc["x"]
    assert linha["p_valor"] < 0.05
    assert bool(linha["distribuicao_mudou"]) is True


def test_comparar_distribuicoes_categoricas_calcula_proporcoes():
    df_2023 = pd.DataFrame({"rede": ["Municipal"] * 80 + ["Estadual"] * 20})
    df_2024 = pd.DataFrame({"rede": ["Municipal"] * 50 + ["Estadual"] * 50})

    resultado = comparar_distribuicoes_categoricas(df_2023, df_2024, ["rede"])
    resultado = resultado.set_index("categoria")

    assert abs(resultado.loc["Municipal", "proporcao_2023"] - 0.8) < 1e-9
    assert abs(resultado.loc["Municipal", "proporcao_2024"] - 0.5) < 1e-9
    assert abs(resultado.loc["Municipal", "diferenca_absoluta"] - 0.3) < 1e-9
