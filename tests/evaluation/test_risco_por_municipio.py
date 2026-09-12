import numpy as np
import pandas as pd

from src.evaluation.risco_por_municipio import agregar_risco_por_municipio


def test_agrega_risco_medio_por_municipio_e_ordena_desc():
    id_municipio = pd.Series(["A", "A", "B", "B", "B"])
    probabilidades = np.array([0.2, 0.4, 0.9, 0.8, 0.7])

    resultado = agregar_risco_por_municipio(id_municipio, probabilidades)

    assert list(resultado["id_municipio"]) == ["B", "A"]
    assert abs(resultado.set_index("id_municipio").loc["A", "risco_medio"] - 0.3) < 1e-9
    assert resultado.set_index("id_municipio").loc["B", "n_alunos"] == 3
