import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X=iris.data
y=iris.target

scaler=StandardScaler()
X=scaler.fit_transform(X)

X=torch.tensor(X, dtype=torch.float32)
y=torch.tensor(y, dtype=torch.long)

x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)

class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()   
        self.hidden=nn.Linear(4, 16)
        self.output=nn.Linear(16, 3)
        
    def forward(self, x):
        x=torch.relu(self.hidden(x))
        x=self.output(x)
        return x
    
model= IrisNN()
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    output=model(x_train)
    loss=criterion(output, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(loss.item())
        
entrada= x_test[0].reshape(1,-1)
print (entrada)

with torch.no_grad():
  salida= model(entrada)
  print(salida)
  i, y_pred= torch.max(salida, 1)
  print(y_pred)
  print(iris.target_names[y_pred])
  
  y_test[0]
print(iris.target_names[y_test[0]])