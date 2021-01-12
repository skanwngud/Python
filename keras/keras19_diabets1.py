import numpy as np

from sklearn.datasets import load_diabetes

dataset=load_diabetes()
x=dataset.data
y=dataset.target

# print(x[:5])
# print(y[:10])
# print(x.shape) # (442, 10)
# print(y.shape) # (442, )

# print(np.max(x), np.min(y))
# print(dataset.feature_names)
# print(dataset.DESCR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

input1=Input(shape=10)
dense1=Dense(120, activation='relu')(input1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
output=Dense(1, activation='relu')(dense1)
model=Model(inputs=input1, outputs=output)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=130, batch_size=6, validation_data=(x_val, y_val))

loss, mae=model.evaluate(x_test, y_test, batch_size=1)
pred=model.predict(x_test)

def RMSE(y_test, pred):
    return np.sqrt(mean_squared_error(y_test, pred))

rmse=RMSE(y_test, pred)
r2=r2_score(y_test, pred)

print('loss : ', loss)
print('mae : ', mae)
print('RMSE : ', rmse)
print('r2 : ', r2)

# loss :  6393.5126953125
# mae :  64.18359375
# RMSE :  79.95945517502678
# r2 :  0.014872931618142626