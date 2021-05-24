import torch

tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(tensor1)

tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])
print(tensor2)

# tensor 의 사칙연산
add_tensor = tensor1 + tensor2
print(add_tensor)

sub_tensor = tensor1 - tensor2
print(sub_tensor)

mul_tensor = tensor1 * tensor2
print(mul_tensor)

div_tensor = tensor1 / tensor2
print(div_tensor)


# pytorch 의 제공 모듈
print(torch.add(tensor1, tensor2))
print(torch.sub(tensor1, tensor2))
print(torch.mul(tensor1, tensor2))
print(torch.div(tensor1, tensor2))

print(torch.matmul(tensor1, tensor2))