"""Definição de features e pré-processamento integrado (SimpleImputer +
OneHotEncoder dentro de um ColumnTransformer) — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "taxa_alfabetizacao_ano_anterior",
    "gap_meta_resultado_ano_anterior",
    "meta_alfabetizacao_ano_anterior",
]
CATEGORICAL_FEATURES = ["rede", "regiao", "sigla_uf"]


def preparar_target(df: pd.DataFrame) -> pd.Series:
    """Deriva 'em_risco' (1 = não alfabetizado, 0 = alfabetizado).

    Convenção usada em toda avaliação do projeto: a classe positiva (1)
    é a classe de interesse do negócio (risco), alinhando com o
    comportamento padrão do scikit-learn (predict_proba[:, 1],
    average_precision_score, precision_recall_curve) sem precisar de
    'pos_label' explícito em cada chamada.
    """
    return (1 - df["alfabetizado"]).rename("em_risco")


def build_preprocessor() -> ColumnTransformer:
    """Pré-processamento a ser integrado ao modelo via sklearn Pipeline
    (nunca aplicado manualmente antes do fit — ver Task 4)."""
    transformador_numerico = SimpleImputer(strategy="median", add_indicator=True)
    transformador_categorico = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("numericas", transformador_numerico, NUMERIC_FEATURES),
        ("categoricas", transformador_categorico, CATEGORICAL_FEATURES),
    ])
