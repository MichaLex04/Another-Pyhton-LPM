import torch

x=torch.tensor([[0., 0.],[0., 1.],[1., 0.],[1., 1.]])
y=torch.tensor([[0.],[0.],[0.],[1.]])

w=torch.randn((2,1), requires_grad=True)
b=torch.randn(1, requires_grad=True)

def sigmoid(x):
    return 1/(1+torch.exp(-x))

lr=0.1
epochs=1000

for epoch in range(epochs):
    z=x@w+b
    y_pred=sigmoid(z)
    loss=((y_pred - y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
        w.grad.zero_()
        b.grad.zero_()
    if epoch % 100==0:
        print(epoch-1, loss.item())

z=x@w+b
y_pred=sigmoid(z)
y_pred_boolean=(y_pred>=0.5).float()
print(y_pred_boolean)