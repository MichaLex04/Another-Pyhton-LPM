#Importamos las librerias necesarias
import sklearn as sk
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Cargamos los conjuntos de datos de los vinos
wine=load_wine()
x=wine.data
y=wine.target

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)<

# Crea un modelo de árbol de decisión
model= DecisionTreeClassifier()
model.fit(x_train, y_train)

# Datos del grupo 1(0) de los vinos
y_pred=model.predict([[14.23,  1.71,  2.43,  15.6,  127.0,  2.80,  3.06,  0.28,  2.29,   5.64,  1.04,  3.92,  1065.0]])
print(y_pred)

# Datos del grupo 2(1) de los vinos
y2_pred=model.predict([[13.17,  2.59,  2.37,  20.0,  120.0,  1.65,  0.68,  0.53,  1.46,   9.30,  0.60,  1.62,   840.0]])
print(y2_pred)

wine.target_names[y_pred]


