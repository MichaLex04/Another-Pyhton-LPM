import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Modulo nn de torch para implementar la RNA
#Split de dataser para entrenamiento y pruebas
#Permite escalar los datos de manera numerica(normalizacion)

iris= load_iris()
X = iris.data
y = iris.target

#Obtenemos la data de iris, segmentando datos y etiquetas

scaler=StandardScaler()
X = scaler.fit_transform(X)

#Normalizamos los datos para estabilizar el entrenamiento (media de 0 y desviacion estandar de 1)

X=torch.tensor(X, dtype=torch.float32)
y=torch.tensor(y, dtype=torch.long)

#Convertimos datos normalizados y etiquetas a tensores (Estructura de datos base de PyTorch)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Dividimos los datos en entrenamiento y prueba, con un 20% de datos para prueba

class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        self.hidden = nn.Linear(4, 16)
        self.output = nn.Linear(16, 3)
        
    def forward(self, x):
        x=torch.relu(self.hidden(x))
        x=self.output(x)
        return x
    
model=IrisNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    output = model(x_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(epoch, loss.item())
        
entrada=np.array([[5.1, 3.5, 1.4, 0.2]])
entrada_escalada = scaler.transform(entrada)
entrada_tensor = torch.tensor(entrada_escalada, dtype=torch.float32)

with torch.no_grad():
    salida = model(entrada_tensor)
    i, y_pred = torch.max(salida, 1)
    y_pred
    
iris.target_names[y_pred]