import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train=pd.read_csv('./dacon/train/train.csv', header=0, index_col=0)
sub=pd.read_csv('./dacon/sample_submission.csv', header=0, index_col=0)

df_test = []

for i in range(81):
    file_path = './dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    df_test.append(temp)

x_test = pd.concat(df_test)

# print(x_train.info())
# print(x_test.info())

df_train=x_train.iloc[:, 2:]
df_test=x_test.iloc[:, 3:]

df_test.to_csv('./dacon/dacon_test.csv', sep=',')

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

df_train=df_train.reshape(-1, 48, 6)
df_test=df_test.reshape(-1, 7*48, 6)

def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col-1
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 3)

print(x.shape) # (1087, 7, 48, 6)
print(y.shape) # (1087, 2, 48, 6)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=False)

x_train=x_train.reshape(-1, 7*48, 6)
x_test=x_test.reshape(-1, 7*48, 6)

y_train=y_train.reshape(-1, 2*48, 6)
y_test=y_test.reshape(-1, 2*48, 6)

model=Sequential()
model.add(LSTM(128, activation='relu', input_shape=(7*48, 6)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2*48*6))
model.add(Reshape((96, 6)))

es=EarlyStopping(monitor='val_loss', patience=20, mode='auto')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_data_day_2_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', mode='auto', save_best_only=True)
rl=ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', factor=0.1, min_delta=0.01)
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, validation_split=0.2,
            epochs=100, batch_size=64, callbacks=[es, cp, rl])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(df_test)

print(y_pred[0])
print(y_pred.shape)
print(loss)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss & val_loss')

plt.show()