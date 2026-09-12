"""Candidatos de algoritmo, cada um como Pipeline (pré-processamento +
classificador) — ver docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md,
seção 'Modelo e seleção'."""

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def construir_candidatos(preprocessador: ColumnTransformer) -> dict[str, Pipeline]:
    """Cada Pipeline recebe seu próprio clone do preprocessador — evita que o
    fit de uma pipeline sobrescreva o estado ajustado (mediana, categorias)
    de outra, já que as três compartilhariam a mesma instância em memória
    se `clone()` não fosse usado aqui."""
    return {
        "regressao_logistica": Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
    }
