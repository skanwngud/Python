import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_diabetes

dataset=load_diabetes()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

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

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=130, batch_size=6, validation_data=(x_val, y_val))

loss, mae=model.evaluate(x_test, y_test, batch_size=1)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse=RMSE(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print('loss : ', loss)
print('mae : ', mae)
print('rmse : ', rmse)
print('r2 : ', r2)

# 데이터 전처리 전
# loss :  6393.5126953125
# mae :  64.18359375
# RMSE :  79.95945517502678
# r2 :  0.014872931618142626

# 데이터 전처리 후 - x/422.
# loss :  3795.854736328125
# mae :  51.651878356933594
# rmse :  61.61050908869314
# r2 :  0.4151261630377242

# 데이터 전처리 후 - MinMax x
# loss :  6232.6943359375
# mae :  62.160133361816406
# rmse :  78.94741910881152
# r2 :  0.03965235948904722

# 데이터 전처리 후 - MinMax x_train (validation_split)
# loss :  11557.19140625
# mae :  84.50377655029297
# rmse :  107.50438440532257
# r2 :  -0.7807582011748044

# 데이터 전처리 후 - MinMax x_train (validation_data)
# loss :  5647.31201171875
# mae :  56.992149353027344
# rmse :  75.1486058670798
# r2 :  0.12984933580868452