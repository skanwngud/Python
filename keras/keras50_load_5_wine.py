import numpy as np

x=np.load('../data/npy/wine_x.npy')
y=np.load('../data/npy/wine_y.npy')

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

y=y.reshape(-1, 1)

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

enc=OneHotEncoder()
enc.fit(y)
y=enc.transform(y).toarray()

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=34)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=34)

input1=Input(shape=(x_train.shape[1],))
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='val_loss', patience=20, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=(x_val, y_val),
            callbacks=[early])

loss=model.evaluate(x_test, y_test)
model.predict(x_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

# results
# loss :  2.759888411674183e-05
# acc :  1.0