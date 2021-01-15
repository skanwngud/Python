import numpy as np

x=np.load('../data/npy/cancer_x.npy')
y=np.load('../data/npy/cancer_y.npy')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

y=y.reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(y)
y=enc.transform(y).toarray()

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=33)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=33)

model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=es, verbose=2)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_test : ', np.argmax(y_test[:5], axis=-1))
print('y_pred : ', np.argmax(y_pred[:5], axis=-1))

# results
# loss :  [0.6787706613540649, 0.9649122953414917]
# y_test :  [0 1 1 1 1]
# y_pred :  [0 1 1 1 1]