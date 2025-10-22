import pandas as pd
import numpy as np

alumnos = pd.DataFrame({
    "Id": [1, 2, 3],
    "Nombre":["A", "B", "C"]
})

notas = pd.DataFrame({
    "Id": [1, 2, 3],
    "Nota":["A", "B", "C"]
})

boleta = pd.merge(alumnos, notas, on="Id", how="inner")
boleta["estado"] = np.where(boleta.nota>12,"A","C")
print(boleta)

