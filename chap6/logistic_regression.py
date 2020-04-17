import torch
from torch.autograd import Variable
from torch.nn import functional

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = functional.sigmoid(self.linear(x))
        return y_pred

model = Model()

criteration = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)

    loss = criteration(y_pred, y_data)
    print(epoch, loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = Variable(torch.Tensor([1.0]))
print("predict (after training)", 1, model.forward(hour_var).data > 0.5)
hour_var = Variable(torch.Tensor([7.0]))
print("predict (after training)", 7, model.forward(hour_var).data > 0.5)
