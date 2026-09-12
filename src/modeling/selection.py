"""Seleção de modelo via validação — critério: PR-AUC (average_precision_score)
da classe 'em_risco'. Ver docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md,
seção 'Modelo e seleção'."""

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score


def selecionar_melhor_modelo(
    candidatos: dict[str, Any],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_validacao: pd.DataFrame,
    y_validacao: pd.Series,
) -> tuple[str, Any, pd.DataFrame]:
    resultados = []
    for nome, pipeline in candidatos.items():
        pipeline.fit(X_treino, y_treino)
        probabilidades = pipeline.predict_proba(X_validacao)[:, 1]
        pr_auc = average_precision_score(y_validacao, probabilidades)
        resultados.append({"modelo": nome, "pr_auc_validacao": pr_auc})

    tabela_resultados = (
        pd.DataFrame(resultados)
        .sort_values("pr_auc_validacao", ascending=False)
        .reset_index(drop=True)
    )
    nome_vencedor = tabela_resultados.iloc[0]["modelo"]
    return nome_vencedor, candidatos[nome_vencedor], tabela_resultados
