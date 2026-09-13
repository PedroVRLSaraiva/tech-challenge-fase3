import numpy as np
import pandas as pd

from src.evaluation.risco_por_municipio import (
    agregar_risco_por_municipio,
    comparar_risco_real_previsto_por_municipio,
)


def test_agrega_risco_medio_por_municipio_e_ordena_desc():
    id_municipio = pd.Series(["A", "A", "B", "B", "B"])
    probabilidades = np.array([0.2, 0.4, 0.9, 0.8, 0.7])

    resultado = agregar_risco_por_municipio(id_municipio, probabilidades)

    assert list(resultado["id_municipio"]) == ["B", "A"]
    assert abs(resultado.set_index("id_municipio").loc["A", "risco_medio"] - 0.3) < 1e-9
    assert resultado.set_index("id_municipio").loc["B", "n_alunos"] == 3


def test_comparar_risco_real_previsto_por_municipio_calcula_diferenca():
    id_municipio = pd.Series(["A", "A", "B", "B"])
    y_real = np.array([0, 1, 1, 1])          # A: real 0.5 | B: real 1.0
    probabilidades = np.array([0.2, 0.4, 0.9, 0.7])  # A: previsto 0.3 | B: previsto 0.8

    resultado = comparar_risco_real_previsto_por_municipio(id_municipio, y_real, probabilidades)

    assert set(resultado.columns) == {
        "id_municipio", "risco_previsto", "risco_real", "n_alunos", "diferenca",
    }
    linha_a = resultado.set_index("id_municipio").loc["A"]
    assert abs(linha_a["risco_previsto"] - 0.3) < 1e-9
    assert abs(linha_a["risco_real"] - 0.5) < 1e-9
    assert abs(linha_a["diferenca"] - (0.3 - 0.5)) < 1e-9
    assert linha_a["n_alunos"] == 2

    linha_b = resultado.set_index("id_municipio").loc["B"]
    assert abs(linha_b["risco_previsto"] - 0.8) < 1e-9
    assert abs(linha_b["risco_real"] - 1.0) < 1e-9

    # ordenado por risco previsto, descendente
    assert list(resultado["id_municipio"]) == ["B", "A"]
