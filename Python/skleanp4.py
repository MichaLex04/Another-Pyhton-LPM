import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine

wine=load_wine()

datos=pd.DataFrame(wine.data, columns=wine.feature_names)
x=datos[["alcohol","malic_acid"]]

model=KMeans(n_clusters=3, random_state=42)
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
print("Cantidad de vinos por cluster:")
for cluster, cantidad in sorted(clusters_count.items()):
    print(f"Cluster {cluster}: {cantidad} vinos")

plt.scatter(x[["alcohol"]],x[["malic_acid"]], c=etiquetas)
plt.scatter(centroides[:,0],centroides[:,1], marker="x", color="r")
plt.show()
