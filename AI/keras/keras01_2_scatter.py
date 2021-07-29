import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# 1. data
x=np.arange(1, 11)
y=np.array([1,2,4,3,5,5,7,9,8,11])
print('\n', x, '\n', y)

# 2. modeling
model=Sequential()
model.add(Dense(1, input_shape=(1,)))
# model.add(Dense(10)) - 머신러닝 모델에는 히든레이어가 존재하지 않아 속도는 딥러닝에 비해 빠르다
# model.add(Dense(10))
# model.add(Dense(1))

# 3. compile, fitting
optimizer=RMSprop(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x,y,
        epochs=10)

# 4. evaluate, predict
y_pred=model.predict(x)

# visualize
plt.scatter(x,y) # column 이 여러개인 경우 여러번 찍으면 된다
plt.plot(x,y_pred, color='red')
plt.show()