import torch

vector1 = torch.tensor([1., 2., 3.])
print(vector1)

vector2 = torch.tensor([4., 5., 6.])
print(vector2)

# 벡터(vector) 는 스칼라(scalar)의 나열, shape = (1, 1)

# 벡터의 사칙연산
add_vector = vector1 + vector2
print(add_vector) # tensor([5., 7., 9.])

sub_vector = vector1 - vector2
print(sub_vector) # tensor([-3., -3., -3.])

mul_vector = vector1 * vector2
print(mul_vector) # tensor([4., 10., 18.])

div_vector = vector1 / vector2
print(div_vector) # tensor([0.2500, 0.4000, 0.5000])

# pytorch 에서 제공하는 모듈로도 사용 가능
print(torch.add(vector1, vector2)) # tensor([5., 7., 9.])
print(torch.sub(vector1, vector2)) # tensor([-3., -3., -3.])
print(torch.mul(vector1, vector2)) # tensor([4., 10., 18.])
print(torch.div(vector1, vector2)) # tensor([0.2500, 0.4000, 0.5000])

# 벡터 내적 요소의 연산
print(torch.dot(vector1, vector2)) # tensor(32.)
# 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32