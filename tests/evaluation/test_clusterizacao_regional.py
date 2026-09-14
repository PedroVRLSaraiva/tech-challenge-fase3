import numpy as np
import pandas as pd

from src.evaluation.clusterizacao_regional import clusterizar_municipios, escolher_k_otimo


def _territorio_dois_grupos_obvios(n_por_grupo: int = 15, semente: int = 0) -> pd.DataFrame:
    """3 features numéricas com 2 grupos bem separados por construção —
    qualquer k-means/silhueta razoável deve achar k=2 e separar os grupos
    sem erro."""
    rng = np.random.default_rng(semente)
    grupo_a = rng.normal(loc=0.2, scale=0.02, size=(n_por_grupo, 3))
    grupo_b = rng.normal(loc=0.8, scale=0.02, size=(n_por_grupo, 3))
    valores = np.vstack([grupo_a, grupo_b])

    return pd.DataFrame({
        "id_municipio": [f"m{i}" for i in range(2 * n_por_grupo)],
        "regiao": ["Sul"] * n_por_grupo + ["Norte"] * n_por_grupo,
        "taxa_alfabetizacao": valores[:, 0],
        "gap_meta_resultado": valores[:, 1],
        "meta_alfabetizacao_2024": valores[:, 2],
    })


def test_escolher_k_otimo_encontra_dois_grupos_bem_separados():
    territorio = _territorio_dois_grupos_obvios()
    from sklearn.preprocessing import StandardScaler

    colunas = ["taxa_alfabetizacao", "gap_meta_resultado", "meta_alfabetizacao_2024"]
    X_escalado = StandardScaler().fit_transform(territorio[colunas])

    k_escolhido, tabela_scores = escolher_k_otimo(X_escalado, k_min=2, k_max=6, random_state=42)

    assert k_escolhido == 2
    assert set(tabela_scores.columns) == {"k", "silhueta"}
    assert list(tabela_scores["k"]) == [2, 3, 4, 5, 6]
    # k=2 deve ter a maior silhueta entre os testados (grupos bem separados)
    assert tabela_scores.set_index("k").loc[2, "silhueta"] == tabela_scores["silhueta"].max()


def test_clusterizar_municipios_separa_os_dois_grupos_corretamente():
    territorio = _territorio_dois_grupos_obvios()

    resultado = clusterizar_municipios(territorio, k_min=2, k_max=6, random_state=42)

    assert "clusters" in resultado and "crosstab_regiao_cluster" in resultado and "k_escolhido" in resultado
    assert resultado["k_escolhido"] == 2

    clusters = resultado["clusters"]
    assert set(clusters.columns) >= {"id_municipio", "regiao", "cluster"}
    assert len(clusters) == len(territorio)

    # Os dois grupos construídos (id m0..m14 vs m15..m29) devem cair em
    # clusters diferentes.
    cluster_grupo_a = set(clusters[clusters["id_municipio"].isin([f"m{i}" for i in range(15)])]["cluster"])
    cluster_grupo_b = set(clusters[clusters["id_municipio"].isin([f"m{i}" for i in range(15, 30)])]["cluster"])
    assert len(cluster_grupo_a) == 1
    assert len(cluster_grupo_b) == 1
    assert cluster_grupo_a != cluster_grupo_b


def test_clusterizar_municipios_crosstab_bate_com_regiao_construida():
    territorio = _territorio_dois_grupos_obvios()
    resultado = clusterizar_municipios(territorio, k_min=2, k_max=6, random_state=42)

    crosstab = resultado["crosstab_regiao_cluster"]
    # Grupo A é 100% "Sul", grupo B é 100% "Norte" -> cada região deve
    # concentrar toda sua contagem num único cluster.
    assert (crosstab.loc["Sul"] > 0).sum() == 1
    assert (crosstab.loc["Norte"] > 0).sum() == 1


def test_clusterizar_municipios_respeita_k_fixo_quando_informado():
    territorio = _territorio_dois_grupos_obvios()
    resultado = clusterizar_municipios(territorio, k=3, random_state=42)

    assert resultado["k_escolhido"] == 3
    assert resultado["clusters"]["cluster"].nunique() == 3
