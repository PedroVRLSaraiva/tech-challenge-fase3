import numpy as np
import pandas as pd

from src.modeling.selection import selecionar_melhor_modelo


class _PipelineFalsa:
    """Duck-type mínimo de uma Pipeline treinada: só precisa de fit/predict_proba.
    Usada para testar a LÓGICA de seleção isolada do comportamento real de
    um algoritmo de ML."""

    def __init__(self, probabilidades_fixas):
        self._probabilidades_fixas = np.asarray(probabilidades_fixas)
        self.foi_treinada = False

    def fit(self, X, y):
        self.foi_treinada = True
        return self

    def predict_proba(self, X):
        p = self._probabilidades_fixas
        return np.column_stack([1 - p, p])


def test_selecionar_melhor_modelo_escolhe_maior_pr_auc_na_validacao():
    y_validacao = pd.Series([0, 0, 1, 1, 1])

    candidatos = {
        "perfeito": _PipelineFalsa([0.01, 0.02, 0.9, 0.95, 0.99]),   # combina com y quase exatamente
        "aleatorio": _PipelineFalsa([0.5, 0.5, 0.5, 0.5, 0.5]),      # não discrimina nada
    }

    X_treino = pd.DataFrame({"x": range(10)})
    y_treino = pd.Series([0, 1] * 5)
    X_validacao = pd.DataFrame({"x": range(5)})

    nome_vencedor, pipeline_vencedora, tabela = selecionar_melhor_modelo(
        candidatos, X_treino, y_treino, X_validacao, y_validacao,
    )

    assert nome_vencedor == "perfeito"
    assert pipeline_vencedora is candidatos["perfeito"]
    assert candidatos["perfeito"].foi_treinada
    assert candidatos["aleatorio"].foi_treinada  # todos os candidatos são treinados

    assert list(tabela["modelo"]) == ["perfeito", "aleatorio"]
    assert tabela.iloc[0]["pr_auc_validacao"] > tabela.iloc[1]["pr_auc_validacao"]
