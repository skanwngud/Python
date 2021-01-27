import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score # 분류 모델에서의 지표. 회귀 모델에서는 R2_score 를 쓴다

# 1. data
x_data=[[0,0], [1, 0], [0, 1], [1, 1]] # (4, 2)
y_data=[0, 1, 1, 0] # (4, )

# 2. modeling
# model=LinearSVC()
# model=SVC()
model=Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# 3. compile, fitting
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data,
            epochs=100, batch_size=1)

# 4. score, predict
results=model.evaluate(x_data, y_data) # loss, acc
y_pred=model.predict(x_data)

print(x_data, '의 예측결과 : ', y_pred)
print('model.evaluate : ', results[1]) # results[0] : loss

# acc=accuracy_score(y_data, y_pred) # y의 실제값과 예측값을 비교
# print('accuracy_score', acc)

# results - LinearSVC
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  [0 0 0 0]
# model.score :  0.5
# accuracy_score 0.5

# results - SVC
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  [0 1 1 0]
# model.score :  1.0
# accuracy_score 1.0

# results - Sequential (no hidden layers)
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  [[0.51698136]
#  [0.58864456]
#  [0.26518965]
#  [0.3254683 ]]
# model.evaluate :  0.5