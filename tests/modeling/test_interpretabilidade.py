import numpy as np
import pandas as pd

from src.preprocessing.features import build_preprocessor
from src.modeling.candidates import construir_candidatos
from src.modeling.interpretabilidade import calcular_shap_values


def _dados(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": rng.normal(70, 10, n),
        "gap_meta_resultado_ano_anterior": rng.normal(0, 5, n),
        "meta_alfabetizacao_ano_anterior": rng.normal(80, 5, n),
        "rede": rng.choice(["Municipal", "Estadual"], n),
        "regiao": rng.choice(["Sudeste", "Nordeste"], n),
        "sigla_uf": rng.choice(["SP", "BA"], n),
    })
    y = rng.integers(0, 2, n)
    return X, y


def test_calcular_shap_values_2d_para_cada_candidato():
    X, y = _dados()
    candidatos = construir_candidatos(build_preprocessor())

    for nome, pipeline in candidatos.items():
        pipeline.fit(X, y)
        valores_shap = calcular_shap_values(pipeline, X.iloc[:10])

        assert valores_shap.values.ndim == 2, nome
        assert valores_shap.values.shape[0] == 10, nome
        assert valores_shap.feature_names is not None, nome
        assert len(valores_shap.feature_names) == valores_shap.values.shape[1], nome
