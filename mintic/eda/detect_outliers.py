import numpy as np
import pandas as pd

data = pd.read_csv('~/Mineria/mintic-base/data/IMDb_Top_700_Movies_2026.csv')


def detect_outliers(data, method='iqr', threshold=1.5):
    # Normalizar entrada a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data)

    resultado = pd.DataFrame(index=data.index)

    for col in data.columns:
        col_data = data[col]                
        mask_validos = col_data.notna()
        validos = col_data[mask_validos]
        n = mask_validos.sum()

        # columnas numericas unicamente
        columnas_numericas = data.select_dtypes(include=[np.number]).columns

    for col in columnas_numericas:
        col_data = data[col]
        mask_validos = col_data.notna()
        validos = col_data[mask_validos]
        n = mask_validos.sum()
        if n == 0:
            resultado[col] = False
            continue

        if method == 'iqr':
            Q1 = validos.quantile(0.25)
            Q3 = validos.quantile(0.75)
            iqr = Q3 - Q1

            Li = Q1 - threshold * iqr
            Ls = Q3 + threshold * iqr

            col_outliers = (col_data < Li) | (col_data > Ls)

        elif method == 'zscore':
            mu = validos.sum() / n
            sig = np.sqrt(((validos - mu) ** 2).sum() / n)

            if sig == 0:
                col_outliers = pd.Series(False, index=data.index)
            else:
                Z = (col_data - mu) / sig
                col_outliers = Z.abs() > threshold

        else:
            raise ValueError("El método debe ser 'iqr' o 'zscore'")

        # Forzar False donde el valor original era NaN
        col_outliers = col_outliers.where(mask_validos, False)

        resultado[col] = col_outliers

    return resultado