import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

datos=pd.DataFrame(cancer.data, columns=cancer.feature_names)
x=datos[["mean radius","mean texture"]]

model=KMeans(n_clusters=2, random_state=42)
model.fit(x)

centroides=model.cluster_centers_
etiquetas=model.labels_

datos1=np.array(x)
clusters={}

for punto, etiqueta in zip(datos1,etiquetas):
    clusters.setdefault(etiqueta,[]).append(punto.tolist())
print(clusters)

clusters_count = {}
for etiqueta in etiquetas:
    clusters_count[etiqueta] = clusters_count.get(etiqueta, 0) + 1
print("Cantidad de pacientes por cluster:")
for cluster, cantidad in sorted(clusters_count.items()):
    print(f"Cluster {cluster}: {cantidad} pacientes")

plt.scatter(x[["mean radius"]],x[["mean texture"]], c=etiquetas)
plt.scatter(centroides[:,0],centroides[:,1], marker="x", color="r")
plt.show()
