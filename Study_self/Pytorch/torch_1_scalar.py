import torch

scalar1 = torch.tensor([1.])
print(scalar1) # tensor([1.])

scalar2 = torch.tensor([3.])
print(scalar2) # tensor([3.])

# 스칼라(scalar)는 쉽게 말해 상수값 shape = 1,

# 스칼라의 사칙연산
add_scalar = scalar1 + scalar2
print(add_scalar) # tensor([4.])

sub_scalar = scalar1 - scalar2
print(sub_scalar) # tensor([-2.])

mul_scalar = scalar1 * scalar2
print(mul_scalar) # tensor([3.])

div_scalar = scalar1 / scalar2
print(div_scalar) # tensor([0.3333])

# pytorch 에서 제공하는 모듈로도 사용 가능
print(torch.add(scalar1, scalar2)) # tensor([4.])
print(torch.sub(scalar1, scalar2)) # tensor([-2.])
print(torch.mul(scalar1, scalar2)) # tensor([3.])
print(torch.div(scalar1, scalar2)) # tensor([0.3333])