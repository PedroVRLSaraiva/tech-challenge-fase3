import numpy as np
import pandas as pd

from src.preprocessing.features import build_preprocessor
from src.modeling.candidates import construir_candidatos


def _dados(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": rng.normal(70, 10, n),
        "gap_meta_resultado_ano_anterior": rng.normal(0, 5, n),
        "meta_alfabetizacao_ano_anterior": rng.normal(80, 5, n),
        "rede": rng.choice(["Municipal", "Estadual"], n),
        "regiao": rng.choice(["Sudeste", "Nordeste"], n),
        "sigla_uf": rng.choice(["SP", "BA"], n),
    })


def test_construir_candidatos_retorna_as_tres_pipelines_esperadas():
    candidatos = construir_candidatos(build_preprocessor())
    assert set(candidatos) == {"regressao_logistica", "random_forest", "hist_gradient_boosting"}
    for pipeline in candidatos.values():
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict_proba")


def test_cada_pipeline_treina_e_preve_sem_erro():
    X = _dados()
    y = np.random.default_rng(1).integers(0, 2, len(X))
    candidatos = construir_candidatos(build_preprocessor())

    for nome, pipeline in candidatos.items():
        pipeline.fit(X, y)
        probabilidades = pipeline.predict_proba(X)
        assert probabilidades.shape == (len(X), 2), nome


def test_pipelines_nao_compartilham_estado_do_preprocessador():
    """Regressão: usar a MESMA instância de ColumnTransformer em 3 Pipelines
    sem clone() faz o fit de uma sobrescrever o estado das outras."""
    preprocessador_base = build_preprocessor()
    candidatos = construir_candidatos(preprocessador_base)

    X_a = _dados(n=30, seed=10)
    X_a["taxa_alfabetizacao_ano_anterior"] = 20.0  # mediana bem distinta de X_b
    y_a = np.random.default_rng(10).integers(0, 2, len(X_a))

    X_b = _dados(n=30, seed=20)
    X_b["taxa_alfabetizacao_ano_anterior"] = 90.0
    y_b = np.random.default_rng(20).integers(0, 2, len(X_b))

    candidatos["regressao_logistica"].fit(X_a, y_a)
    mediana_apos_fit_a = (
        candidatos["regressao_logistica"]
        .named_steps["preprocessamento"]
        .named_transformers_["numericas"]
        .statistics_[0]
    )

    candidatos["random_forest"].fit(X_b, y_b)
    mediana_lida_de_novo_em_a = (
        candidatos["regressao_logistica"]
        .named_steps["preprocessamento"]
        .named_transformers_["numericas"]
        .statistics_[0]
    )

    assert mediana_apos_fit_a == mediana_lida_de_novo_em_a
    assert mediana_apos_fit_a == 20.0
