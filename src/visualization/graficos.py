"""Gráficos de suporte para o notebook de modelagem e para o README."""

import matplotlib
matplotlib.use("Agg")  # evita exigir display gráfico em teste/CI

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler


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


def plot_clusters_municipios(clusters, colunas_features):
    """Projeta os indicadores (>=2 dimensões) em 2D via PCA só para permitir
    visualização — a clusterização em si roda no espaço original das
    features, não nesse espaço reduzido. Cor = cluster (k-means), marcador =
    região oficial: se cor e marcador aparecerem misturados no gráfico, os
    clusters cortam fronteiras regionais; se cada região virar um bloco de
    uma cor só, clusters e região coincidem."""
    X_escalado = StandardScaler().fit_transform(clusters[colunas_features])
    coordenadas = PCA(n_components=2, random_state=42).fit_transform(X_escalado)

    clusters = clusters.copy()
    clusters["_pca1"] = coordenadas[:, 0]
    clusters["_pca2"] = coordenadas[:, 1]

    marcadores_disponiveis = ["o", "s", "^", "D", "P", "X", "v", "*"]
    regioes = sorted(clusters["regiao"].unique())
    mapa_marcador = {regiao: marcadores_disponiveis[i % len(marcadores_disponiveis)] for i, regiao in enumerate(regioes)}

    fig, eixo = plt.subplots(figsize=(7, 6))
    for regiao, grupo in clusters.groupby("regiao"):
        eixo.scatter(
            grupo["_pca1"], grupo["_pca2"], c=grupo["cluster"], cmap="tab10",
            vmin=clusters["cluster"].min(), vmax=clusters["cluster"].max(),
            marker=mapa_marcador[regiao], label=regiao, edgecolor="black", linewidth=0.3,
        )
    eixo.set_xlabel("Componente principal 1")
    eixo.set_ylabel("Componente principal 2")
    eixo.set_title("Clusters de municípios (cor) vs. região oficial (marcador)")
    eixo.legend(title="Região", loc="best", fontsize=8)
    fig.tight_layout()
    return fig
