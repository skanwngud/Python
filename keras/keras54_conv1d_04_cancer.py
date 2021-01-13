import numpy as np

x=np.load('../data/npy/cancer_x.npy')
y=np.load('../data/npy/cancer_y.npy')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Input, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x=x.reshape(x.shape[0], x.shape[1], 1)
y=to_categorical(y)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=12)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=12)

print(x_train.shape) # (364, 30, 1)
print(y_train.shape) # (364, 30)

input=Input(shape=(x_train.shape[1], 1))
cnn1=Conv1D(100, 2, padding='same')(input)
max1=MaxPooling1D(2)(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv1D(100, 2, padding='same')(drop1)
max1=MaxPooling1D(2)(cnn1)
drop1=Dropout(0.2)(max1)
flat1=Flatten()(drop1)
dense1=Dense(100, activation='relu')(flat1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
output=Dense(2, activation='sigmoid')(dense1)
model=Model(input, output)

# input=Input(shape=(x.shape[1], 1))
# lstm1=LSTM(100, activation='relu')(input)
# dense1=Dense(100, activation='relu')(lstm1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(100, activation='relu')(dense1)
# dense1=Dense(80, activation='relu')(dense1)
# output=Dense(2, activation='sigmoid')(dense1)
# model=Model(input, output)

es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=es)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_pred[:5], axis=-1))
print(np.argmax(y_test[:5], axis=-1))

# result - conv1D
# [0.2791339159011841, 0.9473684430122375]
# [0 1 1 1 1]
# [0 1 1 1 1]

# result - lstm
# [0.1287907063961029, 0.9385964870452881]
# [0 1 1 1 1]
# [0 1 1 1 1]