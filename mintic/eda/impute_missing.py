import pandas as pd

data = pd.read_csv('~/Mineria/mintic-base/data/IMDb_Top_700_Movies_2026.csv')

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
            
            # Asignamos la estrategia a usar para ESTA columna en particular
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
                        # Corregido: se agregó [mitad] para extraer el valor exacto
                        valor = valores_ordenados[mitad] 

                elif estrategia_actual == 'mode':
                    frecuencias = {}
                    for v in valores_validos:
                        frecuencias[v] = frecuencias.get(v, 0) + 1
                    valor = max(frecuencias, key=frecuencias.get)    
            
            if valor is not None:
                df[col] = df[col].fillna(valor)
    return df
