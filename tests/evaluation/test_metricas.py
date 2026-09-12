import numpy as np

from src.evaluation.metricas import calcular_metricas_teste, tabela_limiares


class _PipelineFalsa:
    def __init__(self, probabilidades):
        self._probabilidades = np.asarray(probabilidades)

    def predict_proba(self, X):
        p = self._probabilidades
        return np.column_stack([1 - p, p])


def test_calcular_metricas_teste_com_separacao_perfeita():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])
    pipeline = _PipelineFalsa(probabilidades)

    metricas = calcular_metricas_teste(pipeline, X=None, y=y)

    assert metricas["roc_auc"] == 1.0
    assert metricas["pr_auc"] == 1.0


def test_tabela_limiares_encontra_o_limiar_perfeito_para_f1():
    # y separa perfeitamente em 0.4: scores < 0.4 -> 0, scores >= 0.4 -> 1
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    probabilidades = np.array([0.1, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 0.9])
    pipeline = _PipelineFalsa(probabilidades)

    tabela = tabela_limiares(pipeline, X=None, y=y, recall_alvo=0.8)

    linha_f1 = tabela.set_index("cenario").loc["otimizado para F1"]
    assert linha_f1["precisao"] == 1.0
    assert linha_f1["recall"] == 1.0
    assert linha_f1["limiar"] == 0.4

    linha_recall_alvo = tabela.set_index("cenario").loc["recall-prioritario (recall >= 80%)"]
    assert linha_recall_alvo["recall"] >= 0.8

    linha_padrao = tabela.set_index("cenario").loc["padrao (limiar 0.5)"]
    assert linha_padrao["limiar"] == 0.5
