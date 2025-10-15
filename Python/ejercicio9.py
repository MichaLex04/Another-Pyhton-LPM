import torch
import torch.nn as nn

x=torch.tensor([[0., 0.],[0., 1.],[1., 0.],[1., 1.]])
y=torch.tensor([[0.],[0.],[0.],[1.]])

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    
model=SimpleNN()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

for epoch in range(850):
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        print(epoch, loss.item())
        print(output.round())