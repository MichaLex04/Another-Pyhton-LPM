import torch
import torch.nn as nn
import numpy as np

x1=np.linspace(-10,10,100).reshape(-1,1)
y1=np.array([1 if i[0] > 0 else 0 for i in x1])
x=torch.tensor(x1,dtype=torch.float32)
y=torch.tensor(y1,dtype=torch.float32).reshape(-1,1)

class SimpleNN(nn.Module):
    def __init__ (self):
        super(SimpleNN, self).__init__()
        self.hidden= nn.Linear(1, 3)
        self.output= nn.Linear(3, 1)
        self.sigmoid= nn.Sigmoid()
        
    def forward(self, x):
        x=self.sigmoid(self.hidden(x))
        x=self.sigmoid(self.output(x))
        return x
    
model=SimpleNN()
criterion= nn.BCELoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    outputs= model(x)
    loss= criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward( )
    optimizer.step()
    
    if epoch % 10==0:
        print(loss . item())
        print(outputs.round())
    
with torch.no_grad():
    y_pred= model(torch.tensor([[50.]]))
    print(y_pred.round())