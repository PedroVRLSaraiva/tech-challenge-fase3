"""SHAP values do modelo vencedor — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md, seção
'Interpretabilidade'.

Nota: shap.TreeExplainer retorna formato diferente conforme o modelo —
RandomForestClassifier devolve um array 3D (amostras, features, classes),
enquanto HistGradientBoostingClassifier já devolve 2D. Este módulo
normaliza os dois casos para sempre retornar 2D (classe 'em_risco' = 1)."""

import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def calcular_shap_values(pipeline, X_amostra: pd.DataFrame) -> shap.Explanation:
    preprocessador = pipeline.named_steps["preprocessamento"]
    classificador = pipeline.named_steps["classificador"]

    X_transformado = preprocessador.transform(X_amostra)
    nomes_features = list(preprocessador.get_feature_names_out())

    if isinstance(classificador, LogisticRegression):
        explicador = shap.LinearExplainer(classificador, X_transformado)
    else:
        explicador = shap.TreeExplainer(classificador)

    valores_shap = explicador(X_transformado)
    if valores_shap.values.ndim == 3:
        valores_shap = valores_shap[..., 1]  # classe 1 = em_risco

    valores_shap.feature_names = nomes_features
    return valores_shap
