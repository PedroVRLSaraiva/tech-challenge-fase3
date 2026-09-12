"""Métricas de teste e tabela de limiares nomeados — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md, seção
'Avaliação e relato de métricas'. Convenção: y usa 'em_risco' (1 = classe
de interesse)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def calcular_metricas_teste(pipeline, X, y) -> dict:
    probabilidades = pipeline.predict_proba(X)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, probabilidades),
        "pr_auc": average_precision_score(y, probabilidades),
    }


def _precisao_recall_em_limiar(y, probabilidades, limiar: float) -> tuple[float, float]:
    predito = (probabilidades >= limiar).astype(int)
    y = np.asarray(y)
    verdadeiro_positivo = int(((predito == 1) & (y == 1)).sum())
    falso_positivo = int(((predito == 1) & (y == 0)).sum())
    falso_negativo = int(((predito == 0) & (y == 1)).sum())
    precisao = verdadeiro_positivo / (verdadeiro_positivo + falso_positivo) if (verdadeiro_positivo + falso_positivo) else 0.0
    recall = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo) if (verdadeiro_positivo + falso_negativo) else 0.0
    return precisao, recall


def tabela_limiares(pipeline, X, y, recall_alvo: float = 0.8) -> pd.DataFrame:
    probabilidades = pipeline.predict_proba(X)[:, 1]
    precisao, recall, limiares = precision_recall_curve(y, probabilidades)
    # precisao/recall têm 1 elemento a mais que limiares (o último ponto, sem
    # limiar correspondente, representa 'classificar tudo como negativo').
    precisao, recall = precisao[:-1], recall[:-1]

    f1 = 2 * precisao * recall / np.clip(precisao + recall, 1e-12, None)
    idx_f1 = int(np.argmax(f1))

    indices_recall_ok = np.where(recall >= recall_alvo)[0]
    # Entre os limiares que ainda satisfazem o recall mínimo, o de maior
    # índice é o de maior limiar (mais restritivo), que maximiza a precisão
    # sem violar o piso de recall.
    idx_recall_alvo = int(indices_recall_ok.max()) if len(indices_recall_ok) > 0 else int(np.argmax(recall))

    precisao_padrao, recall_padrao = _precisao_recall_em_limiar(y, probabilidades, 0.5)

    return pd.DataFrame([
        {"cenario": "padrao (limiar 0.5)", "limiar": 0.5, "precisao": precisao_padrao, "recall": recall_padrao},
        {"cenario": "otimizado para F1", "limiar": float(limiares[idx_f1]), "precisao": float(precisao[idx_f1]), "recall": float(recall[idx_f1])},
        {
            "cenario": f"recall-prioritario (recall >= {recall_alvo:.0%})",
            "limiar": float(limiares[idx_recall_alvo]),
            "precisao": float(precisao[idx_recall_alvo]),
            "recall": float(recall[idx_recall_alvo]),
        },
    ])
