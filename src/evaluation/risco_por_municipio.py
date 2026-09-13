"""Agregação das predições por município — responde 'quais municípios
apresentam maior risco educacional' (ver spec, seção 'Perguntas de negócio
cobertas')."""

import numpy as np
import pandas as pd


def agregar_risco_por_municipio(id_municipio: pd.Series, probabilidades: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "id_municipio": pd.Series(id_municipio).values,
        "probabilidade_risco": np.asarray(probabilidades),
    })
    agregado = (
        df.groupby("id_municipio")["probabilidade_risco"]
        .agg(risco_medio="mean", n_alunos="count")
        .reset_index()
    )
    return agregado.sort_values("risco_medio", ascending=False).reset_index(drop=True)


def comparar_risco_real_previsto_por_municipio(
    id_municipio: pd.Series, y_real: np.ndarray, probabilidades: np.ndarray,
) -> pd.DataFrame:
    """Compara, por município, o risco médio PREVISTO pelo modelo com a taxa
    REAL observada de em_risco — permite verificar o ranking de risco contra
    o que de fato aconteceu, não só a previsão isolada."""
    df = pd.DataFrame({
        "id_municipio": pd.Series(id_municipio).values,
        "y_real": np.asarray(y_real),
        "probabilidade_risco": np.asarray(probabilidades),
    })
    agregado = (
        df.groupby("id_municipio")
        .agg(
            risco_previsto=("probabilidade_risco", "mean"),
            risco_real=("y_real", "mean"),
            n_alunos=("y_real", "count"),
        )
        .reset_index()
    )
    agregado["diferenca"] = agregado["risco_previsto"] - agregado["risco_real"]
    return agregado.sort_values("risco_previsto", ascending=False).reset_index(drop=True)
