import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

data=pd.DataFrame({
    "nom":["A","B","C","D","E"],
    "len":["JAVA","JS","JAVA","JS","JAVA"],
    "per":["backend","frontend","backend","frontend","fullstack"],
    "exp":[1,3,1,3,5,],
    "sueldo":[1000,3000,1000,3000,5000]
})

x_datos=[["nom","len","per","exp"]]
y_datos=[["sueldo"]]

scaler = StandardScaler()
num_scaler= scaler.fit_transform(data[["exp"]])

encoder = OneHotEncoder(drop="first", sparse_output=False)
cat_encoder= encoder.fit_transform(data[["nom","len","per"]])

final_scaler=np.hstack([cat_encoder, num_scaler])
print(final_scaler)

model=KMeans(n_clusters=3, random_state=42)
model.fit(final_scaler)

centroides=model.cluster_centers_
etiquetas=model.labels_
clusters={}

for punto, etiqueta in zip(final_scaler,etiquetas):
    clusters.setdefault(etiqueta,[]).append(punto.tolist())
print(clusters)

clusters_count = {}
for etiqueta in etiquetas:
    clusters_count[etiqueta] = clusters_count.get(etiqueta, 0) + 1
print("Cantidad de personas por cluster:")
for cluster, cantidad in sorted(clusters_count.items()):
    print(f"Cluster {cluster}: {cantidad} personas")