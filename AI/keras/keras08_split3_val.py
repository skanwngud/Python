from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

x=np.array(range(1,101))
y=np.array(range(1,101))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, shuffle=True)

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model=Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

loss, mae=model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict=model.predict(x_test)
print(y_predict)

# shuffle=False
# loss :  0.07874681800603867
# mae :  0.2774181365966797

# shuffle=True
# loss :  0.01689854823052883
# mae :  0.11417099088430405

# validation = 0.2 (실질적인 train 양이 적어졌기 때문에 평가가 더 낮아질 수 있음)
# loss :  0.0007629828178323805
# mae :  0.024267006665468216
