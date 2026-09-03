import pandas as pd

df = pd.read_csv('~/Mineria/mintic-base/data/IMDb_Top_700_Movies_2026.csv')

def plot_missing(data):
    resultado = {}
    
    for col in data.columns:
        cont = 0
        for val_col in data[col]:
            if val_col is None or val_col != val_col:
                cont += 1
        resultado[col] = cont

    return resultado


