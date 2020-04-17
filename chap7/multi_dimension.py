import torch
from torch.autograd import Variable

raw_x = [
    [2.1, 0.1],
    [4.2, 0.8],
    [3.1, 0.9],
    [3.3, 0.2]
]

x_data = Variable(torch.Tensor(raw_x))
y_data = Variable(torch.Tensor([[0.0], [1.0], [0.0], [1.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
    
    def forward(self, x):
        y_pred = linear(x)
        return y_pred