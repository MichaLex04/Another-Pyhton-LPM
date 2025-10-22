import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
frame = pd.DataFrame(wine.data, columns=wine.feature_names)
frame["target"] = wine.target
frame["target_names"] = wine.target_names[wine.target]

reporte1 = frame["target_names"].value_counts()

reporte2 = frame[frame["target"].isin([0, 2])]["target_names"].value_counts()

plt.bar(reporte1.index, reporte1.values)
plt.title("Distribución de todas las clases")
plt.show()

plt.pie(reporte1, labels=reporte1.index, autopct="%1.1f%%")
plt.title("Distribución porcentual de todas las clases")
plt.show()

plt.bar(reporte2.index, reporte2.values)
plt.title("")
plt.show()