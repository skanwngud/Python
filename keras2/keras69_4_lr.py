x_train=0.5
y_train=0.8

# 실습 - w, l, e 수치 바꿔서 해볼 것
weight=0.5 # 0.3 0.6666
lr=0.01 # 0.1 1 10 0.001
epoch=5 # 10 300 1000

for iteration in range(epoch):
    y_predict=x_train*weight # y 예측값 # 0.25
    error=(y_predict-y_train)**2 # mse # 0.3025

    print('Error : '+ str(error) + '\ty_predict : '+str(y_predict))

    # y 예측값과 error 값을 변수로 저장시킴
    up_y_predict=x_train*(weight+lr) # y 예측값 # 0.255
    up_error=(y_train-up_y_predict)**2 # mse # 0.297025

    down_y_predict=x_train*(weight-lr) # y 예측값 # 0.245
    down_error=(y_train-down_y_predict)**2 # mse # 0.308025

    # weight 값 조정
    if (down_error<=up_error): # 0.297025 <= 0.308025 이므로,
        weight=weight-lr # 값이 커지면 lr 값을 빼줌 # 0.49
    if (down_error>up_error): 
        weight=weight+lr # 값이 작아지면 lr 값을 더해줌
    
    # 1 loop 를 돌아 weight 를 갱신시킴