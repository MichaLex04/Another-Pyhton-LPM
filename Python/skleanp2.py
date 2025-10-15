import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

X = np.array([[100, 50], [120, 60], [400, 200], [380, 220], [700, 300], [720, 320]]) 

model=KMeans(n_clusters=3, random_state=42)
model.fit(X)

centroides=model.cluster_centers_
etiquetas=model.labels_

plt.scatter(X[:,0],X[:,1], c=etiquetas)
plt.scatter(centroides[:,0],centroides[:,1], marker="x", color="r")

clusters={}
for punto, etiqueta in zip(X,etiquetas):
    clusters.setdefault(etiqueta,[].append(punto.tolist()))
print(clusters)