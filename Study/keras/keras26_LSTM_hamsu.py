import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])

x_pred=x_pred.reshape(1,3,1)

input1=Input(shape=(3,1))
LSTM1=LSTM(350, activation='relu')(input1)
dense1=Dense(350, activation='relu')(LSTM1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(350, activation='relu')(dense1)
output1=Dense(1)(dense1)
model=Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

loss=model.evaluate(x, y)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)

# results
# 6.723496426275233e-06
# [[79.527275]]