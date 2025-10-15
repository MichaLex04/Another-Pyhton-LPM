from sklearn.cluster import KMeans
import numpy as np

X=([[1],[2],[3],[4],[7],[8],[9],[10]])

model=KMeans(n_clusters=2, random_state=42)
model.fit(X)

centers=model.cluster_centers_
label=model.labels_

for punto, etiqueta in zip(X,label):
    if etiqueta==1:
        print(punto,"Aprobado")
    else:
        print(punto,"Desaprobado")