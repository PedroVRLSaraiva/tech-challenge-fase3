"""Clusterização de municípios por indicadores territoriais — responde
'quais regiões apresentam padrões semelhantes' comparando os clusters
resultantes (não-supervisionados) com a região oficial do IBGE. Ver
docs/superpowers/specs/2026-09-14-cv-e-clusterizacao-design.md.

Clusteriza municípios, não regiões: com só 5 regiões oficiais, clusterizar
direto sobre elas dá 5 pontos — poucos para um resultado robusto. Clusterizar
municípios e depois cruzar com a região deixa visível se os padrões seguem
(ou cortam) as fronteiras regionais oficiais."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

COLUNAS_INDICADORES = ["taxa_alfabetizacao", "gap_meta_resultado", "meta_alfabetizacao_2024"]


def escolher_k_otimo(X_escalado: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> tuple[int, pd.DataFrame]:
    """Testa k de k_min a k_max e escolhe o de maior silhueta — métrica que
    não exige saber o número 'certo' de clusters de antemão (ao contrário do
    cotovelo/inércia, que exige leitura visual subjetiva)."""
    scores = []
    for k in range(k_min, k_max + 1):
        rotulos = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_escalado)
        scores.append({"k": k, "silhueta": silhouette_score(X_escalado, rotulos)})

    tabela_scores = pd.DataFrame(scores)
    k_escolhido = int(tabela_scores.loc[tabela_scores["silhueta"].idxmax(), "k"])
    return k_escolhido, tabela_scores


def clusterizar_municipios(
    territorio: pd.DataFrame,
    k: int | None = None,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> dict:
    """territorio: 1 linha por município, com id_municipio, regiao e as
    COLUNAS_INDICADORES (ex.: saída de preparar_territorio_por_ano filtrada
    para um ano). Se k=None, escolhe automaticamente via silhueta.

    Retorna um dict com:
    - 'clusters': territorio + coluna 'cluster'
    - 'crosstab_regiao_cluster': contagem de municípios por (região, cluster)
    - 'k_escolhido': k usado
    - 'tabela_silhueta': scores testados (None se k foi fixado manualmente)
    - 'silhueta_final': score de silhueta do clustering final
    """
    X_escalado = StandardScaler().fit_transform(territorio[COLUNAS_INDICADORES])

    tabela_silhueta = None
    if k is None:
        k, tabela_silhueta = escolher_k_otimo(X_escalado, k_min=k_min, k_max=k_max, random_state=random_state)

    rotulos = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_escalado)

    clusters = territorio.copy()
    clusters["cluster"] = rotulos

    crosstab = pd.crosstab(clusters["regiao"], clusters["cluster"])

    return {
        "clusters": clusters,
        "crosstab_regiao_cluster": crosstab,
        "k_escolhido": k,
        "tabela_silhueta": tabela_silhueta,
        "silhueta_final": silhouette_score(X_escalado, rotulos),
    }
