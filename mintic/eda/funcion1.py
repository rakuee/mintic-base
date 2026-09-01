import numpy as np
import pandas as pd

data = pd.read_csv('~/Mineria/mintic-base/data/IMDb_Top_700_Movies_2026.csv')

print("Valores faltantes antes de imputar:")
print(data.isnull().sum())

def impute_missing(data, strategy='mean', columns=None):

    df = data.copy()

    if columns is None:
        Columns_procesar = data.columns
    else:
        Columns_procesar = columns

    for col in Columns_procesar:
            if df[col].isnull().any():
                numeric_column = pd.api.types.is_numeric_dtype(df[col])
            if strategy in ('mean', 'median') and not numeric_column:
                continue
            if strategy == 'mean':
                valor = df[col].mean()

            elif strategy == 'median':
                valor = df[col].median()

            elif strategy == 'mode':
                moda = df[col].mode()
                valor = moda[0] if not moda.empty else None

            if valor is not None:
                df[col] = df[col].fillna(valor)    
    return df

df_imputado = impute_missing(data, strategy='mode', columns=None)

print("\nValores faltantes después de imputar:")
print(df_imputado.isnull().sum())
