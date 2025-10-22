import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

datos=pd.DataFrame({
    "exp": [1,2,3,4,5],
    "rango":["j","j","j","s","s"],
    "contrato":[1,1,1,0,0]
})

x=datos[["exp","rango"]]
y=datos[["contrato"]]

scaler= StandardScaler()
x_num= scaler.fit_transform(x[["exp"]])

encoder=LabelEncoder
x_cat= encoder.fit_transform(x["rango"]).reshape(-1,1)

x=np.stack([x_num,x_cat])
y=np.array(datos[["contrato"]])

print(x_num)