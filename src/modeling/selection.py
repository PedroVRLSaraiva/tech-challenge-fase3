"""Seleção de modelo via validação cruzada — critério: PR-AUC
(average_precision_score) média entre folds da classe 'em_risco'. Ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md, seção
'Modelo e seleção', e docs/superpowers/specs/2026-09-14-cv-e-clusterizacao-design.md.

Substitui a seleção por holdout único (um único split treino/validação) por
k-fold estratificado sobre o pool de desenvolvimento inteiro (treino+
validação fundidos) — reduz a variância da escolha do vencedor, que com um
único split depende de quais linhas caíram em validação por acaso."""

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


def selecionar_melhor_modelo(
    candidatos: dict[str, Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[str, Any, pd.DataFrame]:
    """Para cada candidato, roda StratifiedKFold e mede PR-AUC em cada fold
    (cross_val_score clona o estimador a cada fold — os candidatos originais
    não são mutados nesse passo). O vencedor é escolhido pela média entre
    folds e então refitado no X_dev inteiro, para ser usado como modelo de
    produção (avaliado depois em teste_mesmo_ano/teste_futuro, nunca vistos
    aqui)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    resultados = []
    for nome, pipeline in candidatos.items():
        scores = cross_val_score(pipeline, X_dev, y_dev, cv=cv, scoring="average_precision")
        resultados.append({
            "modelo": nome,
            "pr_auc_cv_media": scores.mean(),
            "pr_auc_cv_desvio": scores.std(),
        })

    tabela_resultados = (
        pd.DataFrame(resultados)
        .sort_values("pr_auc_cv_media", ascending=False)
        .reset_index(drop=True)
    )
    nome_vencedor = tabela_resultados.iloc[0]["modelo"]
    pipeline_vencedora = candidatos[nome_vencedor]
    pipeline_vencedora.fit(X_dev, y_dev)
    return nome_vencedor, pipeline_vencedora, tabela_resultados
