from sklearn.cluster import KMeans
import numpy as np

X=np.array([[1],[2],[3],[8],[9],[10]])

model=KMeans(n_clusters=2, random_state=42)
model.fit(X)

centers= model.cluster_centers_
labels= model.labels_

print("Centroides: ", model.cluster_centers_)
print("Asignaciones: ", model.labels_)

#Declaramos nuestro diccionario de CLUSTERS
#Nuestro bucle itera el array de datos y de etiquetas
#Agregamos a nuestro diccionario cada dato y su etiqueta
"""
clusters={}
for punto, etiqueta in zip(X, labels):
    clusters.setdefault(etiqueta,[]).append(punto.tolist())
print(clusters)
"""

for punto, etiqueta in zip(X,labels):
    if etiqueta==0:
        print(punto,"desaprobado")
    else:
        print(punto,"aprobado")