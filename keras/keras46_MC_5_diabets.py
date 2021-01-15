import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=45)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=45)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

model=Sequential()
model.add(Dense(150, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1))

cp=ModelCheckpoint(filepath='../data/modelcheckpoint/k46_5_diabets_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', save_best_only=True, mode='auto')
early=EarlyStopping(monitor='val_loss', patience=20, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_data=(x_val, y_val), callbacks=[early, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)

# results
# loss :  [2586.006103515625, 39.66070556640625]