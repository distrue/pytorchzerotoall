import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np


class DiabetesDataset(Dataset):
    """
    diabetes dataset
    """
    def __init__(self):
        xy = np.loadtxt('data-diabetes.csv', delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    """
    diabetes model
    """
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.Sigmoid(self.l1(x))
        out2 = self.Sigmoid(self.l2(out1))
        y_pred = self.Sigmoid(self.l3(out2))
        return y_pred

def main():
    dataset = DiabetesDataset()
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    model = Model()

    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # wrap input in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            print(epoch, i, loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

main()