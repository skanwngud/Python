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
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

# train, evaluate
model = Net().to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum=0.5
)
criterion = nn.CrossEntropyLoss()
print(model)

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
            print(f'Train Epoch : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100 * batch_idx/len(train_loader):.0f})%] \tTrain Loss : {loss.item():.4f}')

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

for epoch in range(1, epochs + 1):
    print(f'Epochs : {epoch}\n')
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = evaluate(model, test_loader)
    print(f'\nTest Loss : {test_loss:.4f}, \tAccuracy : {accuracy}\n')