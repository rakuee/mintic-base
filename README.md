# mintic-base

**Tonatiuh Lab**
**ENES Morelia, UNAM**

## Sobre el proyecto

`mintic-base` es el repositorio plantilla sobre el que cada alumno construirá su propia librería de Minería de Datos. El objetivo es comprender el funcionamiento interno de los algoritmos, no únicamente usarlos.

## Reglas del desarrollo

- Toda la lógica de los algoritmos debe implementarse manualmente por el alumno dentro de su subpaquete correspondiente en `mintic/`.
- La única librería numérica permitida para los cálculos del algoritmo es **NumPy**. No está permitido usar `scikit-learn`, `scipy` u otras librerías que ya implementen el algoritmo asignado.
- `pandas` se permite únicamente para la carga y manipulación inicial de datos (lectura de CSV, por ejemplo).
- `matplotlib` se permite para la visualización de resultados.

## Estructura del repositorio

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── sample_dataset.csv
└── mintic/
    ├── __init__.py
    ├── eda/
    │   └── __init__.py
    ├── ensemble/
    │   └── __init__.py
    ├── kmeans/
    │   └── __init__.py
    ├── dbscan/
    │   └── __init__.py
    ├── apriori/
    │   └── __init__.py
    └── pca/
        └── __init__.py
```

Cada subcarpeta dentro de `mintic/` es un subpaquete de Python correspondiente a un reto del curso. El alumno desarrolla su implementación dentro del subpaquete que le sea asignado.


## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

`main.py` carga el dataset de prueba en `data/sample_dataset.csv` y sirve como punto de partida para importar y probar el subpaquete que estés desarrollando.
