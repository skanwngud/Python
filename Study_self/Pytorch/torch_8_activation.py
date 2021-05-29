import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device is {}'.format(device))

batch_size = 32
epochs = 30

# data
train_data = torchvision.datasets.MNIST(
    root = 'data',
    train = True,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.MNIST(
    root = 'data',
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = batch_size
)

# modeling
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout_prob = 0.5

    def forward(self, x):
        x = torch.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

# define model, optimizer, criterion
model = Net().to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum=0.5
)
criterion = nn.CrossEntropyLoss()
print(model)

# train, evaluate
def train(model, train_loader, optimizer, log_interval):
    model.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                'Epochs : {}, \tTrain_Loss : {}/{}({.4f:})%'.format(
                    epoch,
                    batch_idx * len(image),
                    len(train_loader.dataset),
                    100 * batch_idx / len(train_loader),
                ))