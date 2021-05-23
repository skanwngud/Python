# libraries import
import torch

from torch import nn
from torch.utils.data import DataLoader, dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# data
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor(),
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor(),
)

# transfer to DataLoader
batch_size = 64
trian_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print('shape of x [N, C, H, W] : ', x.shape)
    print('shape of y : ', y.shape)
    break

# x, y size
# shape of x [N, C, H, W] :  torch.Size([64, 1, 28, 28])
# shape of y :  torch.Size([64])

# modeling
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512), # (input_shape, output_shape)
            nn.ReLU(),
            nn.Linear(512 , 512), # (prelayer's output(input), output)
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# loss, optimizer define
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y) # y_pred, y_test

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f'loss : {loss:>7f} [{current : 5d}/{size:>5d}]')

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= size
        correct /= size

        print(f'Test Error : \n Accuracy : {100 * correct:>0.1f}%, \
            Avg loss : {test_loss:>8f}')

epochs = 5
for t in range(epochs):
    print(f'Epoch : {t + 1}\n------------------------')
    train(trian_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

print('train done!')

# model save
torch.save(
    model.state_dict(),
    'c:/data/modelcheckpoint/model.pth')
print('saved pytorch model state to model.pth')

# load model
model = NeuralNetwork()
model.load_state_dict(torch.load(
    'c:/data/modelcheckpoint/model.pth'
))

classes = [
    'T-Shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted : "{predicted}", Actual : {actual}')