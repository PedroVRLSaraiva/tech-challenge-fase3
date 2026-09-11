import numpy as np
import pandas as pd

from src.preprocessing.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    preparar_target,
)


def test_preparar_target_inverte_alfabetizado():
    df = pd.DataFrame({"alfabetizado": [1, 0, 1, 1]})
    resultado = preparar_target(df)
    assert list(resultado) == [0, 1, 0, 0]
    assert resultado.name != "alfabetizado"


def _dados_treino():
    return pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": [70.0, np.nan, 80.0, 65.0],
        "gap_meta_resultado_ano_anterior": [-5.0, -2.0, np.nan, 3.0],
        "meta_alfabetizacao_ano_anterior": [80.0, 80.0, 80.0, 80.0],
        "rede": ["Municipal", "Estadual", "Municipal", "Privada"],
        "regiao": ["Sudeste", "Sudeste", "Nordeste", "Sul"],
        "sigla_uf": ["SP", "SP", "BA", "PR"],
    })


def test_build_preprocessor_imputa_nulos_e_sinaliza_ausencia():
    preprocessador = build_preprocessor()
    transformado = preprocessador.fit_transform(_dados_treino())

    assert not np.isnan(transformado).any()

    nomes = list(preprocessador.get_feature_names_out())
    assert any("missingindicator" in nome for nome in nomes)


def test_build_preprocessor_ignora_categoria_desconhecida_no_transform():
    preprocessador = build_preprocessor()
    preprocessador.fit(_dados_treino())

    dados_novos = pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": [50.0],
        "gap_meta_resultado_ano_anterior": [1.0],
        "meta_alfabetizacao_ano_anterior": [80.0],
        "rede": ["Federal"],  # categoria não vista no treino
        "regiao": ["Norte"],  # categoria não vista no treino
        "sigla_uf": ["AC"],   # categoria não vista no treino
    })

    # Não deve levantar exceção (handle_unknown='ignore').
    transformado = preprocessador.transform(dados_novos)
    assert transformado.shape[0] == 1


def test_numeric_and_categorical_features_batem_com_o_esperado():
    assert NUMERIC_FEATURES == [
        "taxa_alfabetizacao_ano_anterior",
        "gap_meta_resultado_ano_anterior",
        "meta_alfabetizacao_ano_anterior",
    ]
    assert CATEGORICAL_FEATURES == ["rede", "regiao", "sigla_uf"]
