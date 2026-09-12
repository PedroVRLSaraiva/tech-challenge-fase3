import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.graficos import plot_curva_precisao_recall, plot_importancia_features


class _PipelineFalsa:
    def __init__(self, probabilidades):
        self._probabilidades = np.asarray(probabilidades)

    def predict_proba(self, X):
        p = self._probabilidades
        return np.column_stack([1 - p, p])


def test_plot_curva_precisao_recall_retorna_figure():
    y = np.array([0, 0, 1, 1])
    pipeline = _PipelineFalsa([0.1, 0.4, 0.6, 0.9])

    fig = plot_curva_precisao_recall(pipeline, X=None, y=y)

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_importancia_features_limita_ao_top_n():
    nomes = [f"feature_{i}" for i in range(20)]
    importancias = np.arange(20, dtype=float)

    fig = plot_importancia_features(nomes, importancias, top_n=5)

    eixo = fig.axes[0]
    assert len(eixo.patches) == 5
    plt.close(fig)


def test_plot_importancia_features_lida_com_menos_features_que_top_n():
    nomes = ["a", "b", "c"]
    importancias = np.array([1.0, 2.0, 3.0])

    fig = plot_importancia_features(nomes, importancias, top_n=15)

    eixo = fig.axes[0]
    assert len(eixo.patches) == 3
    plt.close(fig)
