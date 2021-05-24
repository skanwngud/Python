import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else :
    device = torch.device('cpu')

Batch_size = 64
Input_size = 1000
Hidden_size = 100
Output_size = 10

