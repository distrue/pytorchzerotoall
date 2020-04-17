import torch
import numpy as np
from torch.autograd import Variable

xy = np.loadtxt('data-diabetes.csv', delimiter=",", dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(2, 4)
        self.l2 = torch.nn.Linear(4, 3)
        self.l3 = torch.nn.Linear(3, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_data):
        out1 = self.sigmoid(self.l1(x_data))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = Model()

criteration = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    # forward pass
    y_pred = model(x_data)

    # compute and print loss
    loss = criteration(y_pred, y_data)
    print(epoch, loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
