import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df_train=pd.read_csv('./dacon/train/train.csv')
df_sub=pd.read_csv('./dacon/sample_submission.csv')

df_test=list()
for i in range(81):
    test=pd.read_csv('./dacon/test/'+str(i)+'.csv')
    df_test.append(test)

df_test=pd.concat(df_test)

print(df_train.shape) # (52560, 9)
print(df_test.shape) # (27216, 9)
print(df_sub.shape) # (7776, 10)

print(df_train.info())
print(df_test.info())
print(df_sub.info())
df_train=df_train.drop(['Day', 'Hour', 'Minute'], axis=1)

df_train=df_train.to_numpy()

df_train=df_train.reshape(-1, 48, 6)

def split_2(dataset, size, col):
    x,y=list(), list()
    for i in range(len(dataset)):
        x_end_number=i+size
        y_end_number=x_end_number+col
        if x_end_number > len(dataset):
            break
        tmp_x, tmp_y=dataset[i:x_end_number], dataset[x_end_number:y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x,y=split_2(df_train, 8, 3)

x=x[:, :-1]
y=y[:]

print('x shape:', x.shape) # (1088, 7, 48, 6)
print('y shape:', y.shape) # (1088, )

x=x.reshape(-1, 7*48*6)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=32)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=32)

ss=StandardScaler()
ss.fit(x_train)
x_train=ss.transform(x_train)
x_test=ss.transform(x_test)
x_val=ss.transform(x_val)
# df_test=ss.transform(df_test)

x_train=x_train.reshape(-1, 7, 48, 6)
x_test=x_test.reshape(-1, 7, 48, 6)
x_val=x_val.reshape(-1, 7, 48, 6)

def models():
    model=Sequential()
    model.add(Conv2D(128, (2,2), padding='same', activation='relu', input_shape=(7, 48, 7)))
    model.add(Conv2D(256, (1,2), padding='same', activation='relu'))
    model.add(Conv2D(512, (1,2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2*48*2))
    model.add(Reshape((2,48,2)))
    model.add(Dense(1))

    return model

quantile_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

es=EarlyStopping(monitor='val_loss', patience=20, mode='min')
rl=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min', factor=0.1)

for q in quantile_list:
    model=models()
    model.compile(loss=lambda y_true, y_pred:quantile_loss(q, y_true, y_pred),
                optimizer='adam', metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=100, batch_size=256, callbacks=[es, rl])

    y_pred=model.predict(df_test)
    y_pred=y_pred.reshape(y_pred.shapep[0]*y_pred.shape[1]*y_pred.shape[2]*y_pred.shape[3])
    y_predict=pd.DataFrame(y_pred)
    y_predict=pd.concat(y_predict)
    y_predict2[y_pred<0]=0

    df_sub.loc[df_sub.id.str.contains('Day7'), 'q_0.1':]=y_predict[:, 0].round(2)
    df_sub.loc[df_sub.id.str.contains('Day8'), 'q_0.2':]=y_predict[:, 1].round(2)

df_sub.to_csv('./dacon/sss.csv')

ranges = 336
hours = range(ranges)
sub=sub[ranges:ranges+ranges]

q_01 = sub['q_0.1'].values
q_02 = sub['q_0.2'].values
q_03 = sub['q_0.3'].values
q_04 = sub['q_0.4'].values
q_05 = sub['q_0.5'].values
q_06 = sub['q_0.6'].values
q_07 = sub['q_0.7'].values
q_08 = sub['q_0.8'].values
q_09 = sub['q_0.9'].values

plt.figure(figsize=(18,2.5))
plt.subplot(1,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='#aa00cc')
plt.plot(hours, q_03, color='#00ccaa')
plt.plot(hours, q_04, color='#ccaa00')
plt.plot(hours, q_05, color='#00aacc')
plt.plot(hours, q_06, color='#aacc00')
plt.plot(hours, q_07, color='#cc00aa')
plt.plot(hours, q_08, color='#000000')
plt.plot(hours, q_09, color='blue')
plt.legend()
plt.show()
