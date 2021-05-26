# 현재 cuda 와 torch version 가 서로 맞지 않기 때문에 cuda 를 잡아주지 않음

import torch

cuda = torch.device('cuda')

print(torch.cuda.is_available())
print(torch.cuda.device_count())
# print(torch.cuda.current_device())
print(torch.cuda.get_device_name())