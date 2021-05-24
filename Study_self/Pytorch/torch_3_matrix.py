import torch

matrix1 = torch.tensor([[1., 2.], [3., 4.]])
print(matrix1)

matrix2 = torch.tensor([[5., 6.], [7., 8.]])
print(matrix2)

# 행렬(matrix) 은 벡터의 나열 (2, 2)

# 행렬의 사칙연산
add_matrix = matrix1 + matrix2
print(add_matrix) # tensor([[6., 8.], [10., 12.]])

sub_matrix = matrix1 - matrix2
print(sub_matrix) # tensor([[-4., -4.], [-4., -4]])

mul_matrix = matrix1 * matrix2
print(mul_matrix) # tensor([[5., 12.], [21., 32.]])

div_matrix = matrix1 / matrix2
print(div_matrix) # tensor([[0.2000, 0.3333], [0.4286, 0.5000]])

# pytorch 에서도 모듈을 제공한다
print(torch.add(matrix1, matrix2)) # tensor([[6., 8.], [10., 12.]])
print(torch.sub(matrix1, matrix2)) # tensor([[-4., -4.], [-4., -4.]])
print(torch.mul(matrix1, matrix2)) # tensor([[5., 12.], [21., 32.]])
print(torch.div(matrix1, matrix2)) # tensor([[0.2000, 0.3333], [0.4286, 0.5000]])

print(torch.matmul(matrix1, matrix2)) # tensor([[19., 22.], [43., 50.]])
# [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = [[19., 22.], [43., 50.]]