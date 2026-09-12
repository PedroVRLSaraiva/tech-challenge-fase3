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
