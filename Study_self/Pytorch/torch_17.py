import torch
import torch.nn as nn
import torch.nn.functional as nn
import torch.nn.init as init

from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

