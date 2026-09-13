import numpy as np

from src.evaluation.metricas import (
    calcular_metricas_teste,
    matriz_confusao_em_limiar,
    tabela_calibracao,
    tabela_limiares,
)


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


def test_matriz_confusao_em_limiar_conta_os_quatro_casos_corretamente():
    # y = [0,0,1,1,1], previsto (limiar 0.5) = [0,1,0,1,1]
    y = np.array([0, 0, 1, 1, 1])
    probabilidades = np.array([0.2, 0.6, 0.3, 0.7, 0.8])

    matriz = matriz_confusao_em_limiar(y, probabilidades, limiar=0.5)

    assert matriz == {
        "verdadeiro_positivo": 2,  # índices 3,4
        "falso_positivo": 1,       # índice 1
        "verdadeiro_negativo": 1,  # índice 0
        "falso_negativo": 1,       # índice 2
    }


def test_tabela_calibracao_reflete_probabilidade_prevista_igual_a_taxa_real():
    # Duas faixas bem separadas: metade das linhas com probabilidade ~0.1
    # e taxa real 0.1 (1 em 10 é em_risco), metade com ~0.9 e taxa real 0.9.
    rng = np.random.default_rng(0)
    n_por_faixa = 200
    y_baixo = rng.binomial(1, 0.1, n_por_faixa)
    y_alto = rng.binomial(1, 0.9, n_por_faixa)
    y = np.concatenate([y_baixo, y_alto])
    probabilidades = np.concatenate([
        np.full(n_por_faixa, 0.1), np.full(n_por_faixa, 0.9),
    ])

    tabela = tabela_calibracao(y, probabilidades, n_faixas=2)

    assert len(tabela) == 2
    assert set(tabela.columns) == {
        "faixa", "probabilidade_media_prevista", "taxa_real_observada", "n_alunos",
    }
    tabela_ordenada = tabela.sort_values("probabilidade_media_prevista").reset_index(drop=True)
    assert tabela_ordenada.loc[0, "probabilidade_media_prevista"] == 0.1
    assert abs(tabela_ordenada.loc[0, "taxa_real_observada"] - 0.1) < 0.06
    assert tabela_ordenada.loc[1, "probabilidade_media_prevista"] == 0.9
    assert abs(tabela_ordenada.loc[1, "taxa_real_observada"] - 0.9) < 0.06
    assert tabela["n_alunos"].sum() == 400
