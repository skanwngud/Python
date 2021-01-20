import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean, maximum

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df_train=pd.read_csv('./dacon/train/train_1.csv', header=0, index_col=0)
df_test=pd.read_csv('./dacon/test/test_1.csv', header=0, index_col=0)
sub=pd.read_csv('./dacon/sample_submission.csv')
# print(df_train.info())
# print(df_test.info())

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

df_train=df_train.reshape(-1, 48, 8)
df_test=df_test.reshape(-1, 7, 48, 6)

def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 2)

# print(x.shape) # (1087, 7, 48, 8)
# print(y.shape) # (1087, 2, 48, 8)
# print(df_test.shape) # (81, 7, 48, 6)

x=x[:, :, :, :6]
y=y[:, :, :, -2:]

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

x_train=x_train.reshape(-1, 7*48*6)
x_test=x_test.reshape(-1, 7*48*6)
df_test=df_test.reshape(-1, 7*48*6)

mms=MinMaxScaler()
mms.fit(x_train)
x_train=mms.transform(x_train)
x_test=mms.transform(x_test)
df_test=mms.transform(df_test)

x_train=x_train.reshape(-1, 7, 48, 6)
x_test=x_test.reshape(-1, 7, 48, 6)
df_test=df_test.reshape(-1, 7, 48, 6)

qunatile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

model=Sequential()
model.add(Conv2D(128, 2, padding='same', activation='relu', input_shape=(7,48,6)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same', activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(48*2*1, activation='relu'))
model.add(Reshape((2, 48, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

es=EarlyStopping(monitor='val_loss', mode='auto', patience=30)
rl=ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=2, factor=0.5, min_delta=0.0001)
cp=ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                filepath='../data/modelcheckpoint/dacon_day_2_2_{epoch:02d}-{val_loss:.4f}.hdf5')
    # model.compile(loss=lambda x_train, y_train:quantile_loss(q, x_train, y_train), optimizer='adam')
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, validation_split=0.2,
            epochs=500, batch_size=64, callbacks=[es, rl])
loss=model.evaluate(x_test, y_test)
pred=model.predict(df_test)

pred=pred.reshape(81*2*48*1)
y_pred=pd.DataFrame(pred)

file_path='./dacon/quantile_loss.csv'
y_pred.to_csv(file_path)
    
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')

# plt.legend(['loss', 'val_loss'])
# plt.show()