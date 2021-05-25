import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else :
    device = torch.device('cpu')

Batch_size = 64     # 파라미터를 업데이트 할 때 계산 되는 데이터 수
Input_size = 1000   # 입력층의 노드 수 (64, 1000)
Hidden_size = 100   # 은닉층의 노드 수 (1000, 100)
Output_size = 10    # 출력층의 노드 수 (100, 10)

x = torch.randn(
    Batch_size,
    Input_size,
    device=device,
    dtype=torch.float,
    requires_grad=False,
) # 입력층으로 들어가는 데이터
  # randn : 평균이 0, 표준편차가 1 인 정규분포에서 샘플링함
  # parameter 값을 업데이트 하기 위해 gradient 를 하는 것이지 input data 를 gradient 하는 것이 아님

y = torch.randn(
    Batch_size,
    Output_size,
    device=device,
    dtype=torch.float,
    requires_grad=False,
) # 출력층을 통해 최종적으로 나가는 데이터

w1 = torch.randn(
    Hidden_size,
    device=device,
    dtype=torch.float,
    requires_grad=True
) # 입력층을 통해 들어온 데이터가 연산 될 은닉층으로 들어가는 데이터

w2 = torch.randn(
    Output_size,
    device=device,
    dtype=torch.float,
    requires_grad=True
) # 은닉층에서 연산이 끝나고 출력층으로 들어가는 데이터

learning_rate = 1e-6
for t in range(1, 501):
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    # mm = 행렬 곱, clamp = torch 에서 제공하는 비선형 함수(relu)

    loss = (y_pred - y).pow(2).sum() # mse
    if t % 100 == 0:
        print('Iteration : ', t, '\t', 'Loss : ', loss.item())
    loss.backward() # 각 파라미터에 gradient 를 계산하고 back propagation 을 진행

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # w 값을 위에서 설정한 learning_rate 를 곱한 뒤 gradient 를 계산한다
        # 음수값을 취한 이유는, 가장 최소값의 gradient 를 찾기 위해 반대방향으로 계산한다

        w1.grad_zero()
        w2.grad_zero()
        # 최종적으로 gradient 가 계산이 되었다면, 0으로 초기화하여 다시 처음부터 반복문을 돌린다

