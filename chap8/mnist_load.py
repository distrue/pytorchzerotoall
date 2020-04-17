# MNIST Dataset
train_dataset = datasets.MNIST(
    root="./data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root="./data/",
    train=False,
    transform=transforms.ToTensor()
)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
