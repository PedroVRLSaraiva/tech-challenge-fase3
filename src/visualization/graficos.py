"""Gráficos de suporte para o notebook de modelagem e para o README."""

import matplotlib
matplotlib.use("Agg")  # evita exigir display gráfico em teste/CI

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def plot_curva_precisao_recall(pipeline, X, y):
    probabilidades = pipeline.predict_proba(X)[:, 1]
    precisao, recall, _ = precision_recall_curve(y, probabilidades)

    fig, eixo = plt.subplots(figsize=(6, 5))
    eixo.plot(recall, precisao)
    eixo.set_xlabel("Recall")
    eixo.set_ylabel("Precisão")
    eixo.set_title("Curva Precisão-Recall (classe: em_risco)")
    fig.tight_layout()
    return fig


def plot_importancia_features(nomes_features, importancias, top_n: int = 15):
    importancias = np.asarray(importancias)
    top_n = min(top_n, len(nomes_features))
    indices_ordenados = np.argsort(importancias)[::-1][:top_n]

    nomes_top = [nomes_features[i] for i in indices_ordenados][::-1]
    valores_top = [importancias[i] for i in indices_ordenados][::-1]

    fig, eixo = plt.subplots(figsize=(8, max(3, top_n * 0.35)))
    eixo.barh(nomes_top, valores_top)
    eixo.set_xlabel("Importância")
    eixo.set_title(f"Top {top_n} features mais importantes")
    fig.tight_layout()
    return fig


def plot_calibracao(tabela_calibracao):
    """Compara probabilidade média prevista com taxa real observada por
    faixa — a linha de referência diagonal (y=x) representa calibração
    perfeita ('quando o modelo diz X% de risco, X% dos casos são risco de
    verdade')."""
    fig, eixo = plt.subplots(figsize=(6, 6))
    eixo.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibração perfeita")
    eixo.plot(
        tabela_calibracao["probabilidade_media_prevista"],
        tabela_calibracao["taxa_real_observada"],
        marker="o", label="Modelo",
    )
    eixo.set_xlabel("Probabilidade média prevista")
    eixo.set_ylabel("Taxa real observada")
    eixo.set_title("Curva de calibração (2024)")
    eixo.legend()
    fig.tight_layout()
    return fig
