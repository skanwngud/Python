# EarlyStopping 을 적용한 모델

from tensorflow.keras.datasets import boston_housing

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

(train_data, train_target), (test_data, test_target)=boston_housing.load_data()

x_train=train_data
y_train=train_target
x_test=test_data
y_test=test_target

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)

input1=Input(shape=13)
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(1, activation='relu')(dense1)
model=Model(inputs=input1, outputs=output1)

callback=EarlyStopping(monitor='loss', patience=10, mode='min')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=500, batch_size=12,
            validation_data=(x_val, y_val), callbacks=callback)

loss, mae=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse=RMSE(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print('loss : ', loss)
print('mae : ', mae)
print('rmse : ', rmse)
print('r2 : ', r2)

# No EarlyStopping
# results
# mse :  11.866742134094238
# mae :  2.380223035812378
# rmse :  3.444813860678402
# r2 :  0.8574460125795427

# EarlyStopping
# results
# loss :  10.597819328308105
# mae :  2.167438268661499
# rmse :  3.2554291754074325
# r2 :  0.8726894622901396