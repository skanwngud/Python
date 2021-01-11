import numpy as np

from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

(x_train, y_train), (x_test, y_test)=boston_housing.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

cp=ModelCheckpoint(filepath='./skanwngud/Study/modelCheckpoint/k46_4_boston_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', save_best_only=True, mode='auto')
early=EarlyStopping(monitor='val_loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=500, batch_size=4, validation_data=(x_val, y_val), callbacks=[early], verbose=2)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ', loss)
print('RMSE : ', RMSE(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))

# results
# loss :  [17.749242782592773, 2.731422185897827]
# RMSE :  4.212985256605526
# R2 :  0.7867801034192484