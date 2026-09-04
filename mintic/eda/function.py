import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np

data = pd.read_csv('~/Mineria/mintic-base/data/IMDb_Top_700_Movies_2026.csv')

def percentil_manual(valores, q):

    ordenados = np.sort(valores.to_numpy())
    n = len(ordenados)
    if n == 0:
        return np.nan
    if n == 1:
        return ordenados[0]

    pos = q * (n - 1)
    piso = int(np.floor(pos))
    techo = int(np.ceil(pos))

    if piso == techo:
        return ordenados[piso]

    fraccion = pos - piso
    return ordenados[piso] + (ordenados[techo] - ordenados[piso]) * fraccion
#--------------------------------------------------------------------------------
def impute_missing(data, strategy='mean', columns=None):
    df = data.copy()

    if columns is None:
        Columns_procesar = data.columns
    else:
        Columns_procesar = columns

    for col in Columns_procesar:
        # Solo procesamos si hay valores nulos
        if df[col].isnull().any():
            numeric_column = pd.api.types.is_numeric_dtype(df[col])
            
            # Asignamos la estrategia 
            estrategia_actual = strategy
            
            # Si se pidió mean o median pero la columna es de texto/categórica, usamos moda
            if strategy in ('mean', 'median') and not numeric_column:
                estrategia_actual = 'mode'

            valores_validos = [x for x in df[col] if pd.notnull(x)]
            n = len(valores_validos)
            valor = None

            if n > 0:
                if estrategia_actual == 'mean':
                    valor = sum(valores_validos) / n

                elif estrategia_actual == 'median':
                    valores_ordenados = sorted(valores_validos)
                    mitad = n // 2

                    if n % 2 == 0:
                        valor = (valores_ordenados[mitad-1] + valores_ordenados[mitad]) / 2.0
                    else:
                        
                        valor = valores_ordenados[mitad] 

                elif estrategia_actual == 'mode':
                    frecuencias = {}
                    for v in valores_validos:
                        frecuencias[v] = frecuencias.get(v, 0) + 1
                    valor = max(frecuencias, key=frecuencias.get)    
            
            if valor is not None:
                df[col] = df[col].fillna(valor)
    return df

#---------------------------------------------------------------------------------------------------


    # Normalizar entrada a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data)

    resultado = pd.DataFrame(index=data.index)

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
            valores_ordenados = np.sort(validos.to_numpy())

            Q1 = percentil_manual(valores_ordenados, 0.25)
            Q3 = percentil_manual(valores_ordenados, 0.75)
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
#----------------------------------------------------------------------------------------------------------------
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
            Q1 = percentil_manual(validos, 0.25)
            Q3 = percentil_manual(validos, 0.75)
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
#----------------------------------------------------------------------------------------------------------------
def handle_outliers(data, method='iqr',action='trim', threshold=1.5):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data)

    data = data.copy

    mascara_outliers = detect_outliers(data, method=method, threshold=threshold)

    if action == 'trim':
        filas_eliminar = mascara_outliers.any(axis=1)
        resultado = data[~filas_eliminar]

    elif action == 'cap':
        resultado = data.copy()
        columnas_numericas = data.select_dtypes(include=[np.number]).columns

        for col in columnas_numericas:
            col_data = data[col]
            mask_validos = col.data.notna()
            validos = col_data[mask_validos]
            n = mask_validos.sum()
            if n == 0:
                continue
            if method == 'iqr':
                Q1 = percentil_manual(validos, 0.25)
                Q3 = percentil_manual(validos, 0.75)
                iqr = Q3 - Q1
                Li = Q1 - threshold * iqr
                Ls = Q3 + threshold * iqr

            elif method == 'zscore':
                mu = validos.sum() / n
                sig = np.sqrt(((validos - mu) ** 2).sum() / n)
                if sig == 0:
                    continue
                Li = mu - threshold * sig
                Ls = mu + threshold * sig

        resultado[col] = col_data.clip(lower=Li, upper=Ls)

    return resultado
#----------------------------------------------------------------------------------------------------------------
def plot_missing(data):
    resultado = {}
    
    for col in data.columns:
        cont = 0
        for val_col in data[col]:
            if val_col is None or val_col != val_col:
                cont += 1
        resultado[col] = cont


    columnas = list(resultado.keys())
    valores = list(resultado.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(columnas, valores)

    plt.xlabel("Columnas")
    plt.ylabel("Cantidad de valores faltantes")
    plt.title("Valores faltantes por columna")
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return resultado