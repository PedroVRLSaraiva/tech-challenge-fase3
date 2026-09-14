import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.selection import selecionar_melhor_modelo


def _dados_separaveis(n_por_classe: int = 30, semente: int = 0):
    """Dataset sintético pequeno mas com sinal real (não fixo por dicionário),
    porque agora a seleção treina de verdade em cada fold — precisa de um
    classificador real para diferenciar 'bom' de 'ruim'."""
    rng = np.random.default_rng(semente)
    x_negativo = rng.normal(loc=-2.0, scale=1.0, size=n_por_classe)
    x_positivo = rng.normal(loc=2.0, scale=1.0, size=n_por_classe)
    X = pd.DataFrame({"x": np.concatenate([x_negativo, x_positivo])})
    y = pd.Series([0] * n_por_classe + [1] * n_por_classe)
    return X, y


def _pipeline_logistica():
    return Pipeline([
        ("escala", StandardScaler()),
        ("classificador", LogisticRegression(random_state=42)),
    ])


def _pipeline_aleatoria():
    """DummyClassifier real (não duck-type) — evita qualquer surpresa com
    sklearn.base.clone(), que cross_val_score usa internamente em cada fold."""
    return Pipeline([
        ("escala", StandardScaler()),
        ("classificador", DummyClassifier(strategy="uniform", random_state=42)),
    ])


def test_selecionar_melhor_modelo_escolhe_o_candidato_com_maior_pr_auc_media_em_cv():
    X_dev, y_dev = _dados_separaveis()

    candidatos = {
        "aleatoria": _pipeline_aleatoria(),
        "logistica": _pipeline_logistica(),
    }

    nome_vencedor, modelo_vencedor, tabela = selecionar_melhor_modelo(
        candidatos, X_dev, y_dev, n_splits=5, random_state=42,
    )

    assert nome_vencedor == "logistica"
    assert modelo_vencedor is candidatos["logistica"]

    assert list(tabela["modelo"]) == ["logistica", "aleatoria"]
    assert tabela.iloc[0]["pr_auc_cv_media"] > tabela.iloc[1]["pr_auc_cv_media"]
    assert {"modelo", "pr_auc_cv_media", "pr_auc_cv_desvio"} == set(tabela.columns)


def test_selecionar_melhor_modelo_refita_o_vencedor_no_pool_inteiro():
    X_dev, y_dev = _dados_separaveis()
    candidatos = {"logistica": _pipeline_logistica()}

    _, modelo_vencedor, _ = selecionar_melhor_modelo(
        candidatos, X_dev, y_dev, n_splits=5, random_state=42,
    )

    # Se foi refitado no pool inteiro, prever sobre o próprio X_dev deve
    # discriminar bem (dataset é linearmente separável por construção).
    probabilidades = modelo_vencedor.predict_proba(X_dev)[:, 1]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y_dev, probabilidades) > 0.95
