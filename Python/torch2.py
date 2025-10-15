import torch
import torch.nn as nn
import numpy as np

x1=np.arange(1,201,1).reshape(100,2)
y1=np.arange(3,400,4).reshape(100,1)

x=torch.tensor(x1,dtype=torch.float32)
y=(x*2)

model=nn.Linear(2,1)

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    outputs=model(x)
    loss=criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch% 10==0:
        print(loss.item())

with torch.no_grad():
    y_pred=model(torch.tensor([[100.,100.]]))
    print(y_pred)