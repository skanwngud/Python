import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

x=np.array([range(100), range(100, 200), range(200, 300), range(300, 400), range(400, 500)]) # (5, 100)
y=np.array([range(500, 600), range(600, 700), range(700, 800)]) # (3, 100)

x=np.transpose(x)
y=np.transpose(y)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=777)

input1=Input(shape=(5,))
dense1=Dense(10)(input1)
dense2=Dense(10)(dense1)
dense3=Dense(10)(dense2)
output=Dense(3)(dense3)
model=Model(inputs=input1, outputs=output)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=130, batch_size=1, validation_split=0.2)

results=model.evaluate(x_test, y_test, batch_size=1)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse=RMSE(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print('mse, mae : ', results)
print('RMSE : ', rmse)
print('r2 : ', r2)