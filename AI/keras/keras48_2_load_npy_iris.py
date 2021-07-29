import numpy as np

x_data=np.load('../data/npy/iris_x.npy')
y_data=np.load('../data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape) # (150, 4)
print(y_data.shape) # (150, )

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

y_data=y_data.reshape(-1, 1)

x_train, x_test, y_train, y_test=train_test_split(x_data, y_data, train_size=0.8, random_state=34)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=34)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

input=Input(shape=(x_train.shape[1],))
dense1=Dense(150, activation='relu')(input)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(300, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
output=Dense(3, activation='softmax')(dense1)
model=Model(input, output)

es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=(x_val, y_val), callbacks=es)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_test[:5], axis=-1))
print(np.argmax(y_pred[:5], axis=-1))

# results
# [0.007472599390894175, 1.0]
# [1 2 0 1 2]
# [1 2 0 1 2]