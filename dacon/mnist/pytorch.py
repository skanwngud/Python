import torch
print(torch.cuda.get_device_name(0)) # 2080
print(torch.cuda.is_available())  # True
print(torch.__version__) # 1.7.1