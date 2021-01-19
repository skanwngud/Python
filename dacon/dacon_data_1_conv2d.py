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

# print(df_train.info())

df_test.to_csv('./dacon/dacon_test.csv', sep=',')

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

# print(df_train.shape) # (52560, 6)

df_train=df_train.reshape(-1, 48, 6)
df_test=df_test.reshape(-1, 7, 48, 6)

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

# print(x[0])

# print(x.shape) # (1087, 7, 48, 6)
# print(y.shape) # (1087, 2, 48, 6)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=False)

# print(x_train.shape) # (695, 7, 48, 6)
# print(x_test.shape) # (218, 7, 48, 6)
# print(x_val.shape) # (174, 7, 48, 6)

# print(y_train.shape) # (659, 2, 48, 6)
# print(y_test.shape) # (218, 2, 48, 6)
# print(y_val.shape) # (174, 2, 48, 6)

x_train=x_train.reshape(-1, 7, 48, 6)
x_test=x_test.reshape(-1, 7, 48, 6)

y_train=y_train.reshape(y_train.shape[0], 2, 48, 6)
y_test=y_test.reshape(-1, 2, 48,6)
# y_val=y_val.reshape(-1, 2, 48,6)

model=Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu',input_shape=(7, 48, 6)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2*48*6, activation='relu'))
model.add(Reshape((2, 48, 6)))

# model=Sequential()
# model.add(LSTM(64, activation='relu', input_shape=(7*48, 6)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(2*48*6))
# model.add(Reshape((96, 6)))

# model.summary()

es=EarlyStopping(monitor='val_loss', patience=100, mode='auto')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_day_2_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', save_best_only=True, mode='auto')
rl=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

optimzer=Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimzer)
hist=model.fit(x_train, y_train, validation_split=0.2,
            epochs=1000, batch_size=64, callbacks=[es, cp, rl], verbose=1)

loss=model.evaluate(x_test, y_test)
pred=model.predict(df_test)

print(loss)
# print(pred[0, :])

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# plt.title('loss, val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.legend(['loss', 'val_loss'])

# plt.show()

pred=pred.reshape(81*2*48, 6)
y_pred=pd.DataFrame(pred)

y_pred.to_csv('./dacon/submission_day2_1.csv', sep=',')