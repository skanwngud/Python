# import library
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train=pd.read_csv('./dacon/train/train.csv') # import train data

# print(train.tail())
# print(train.info()) # (52560, 9)

def preporcess_data(data, is_train=True):
    temp=data.copy()
    temp=temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train==True:
        temp['Target1']=temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2']=temp['TARGET'].shift(-48*2).fillna(method='ffill')
        return temp.iloc[:-96]

    elif is_train==False:
        temp=temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :]

df_train=preporcess_data(train) # add to columns - Target1,2

# print(df_train.iloc[:48]) # print 0 day data
# print(train.iloc[48:96]) # print 1 day data
# print(train.iloc[48+48:96+48]) # print 2 day data
# print(df_train.tail())
# print(df_train.info()) # (52464, 9)

df_test=[] # create df_test list

for i in range(81): # open seires 0~80 test dataset
    file_path='./dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    temp=preporcess_data(temp, is_train=False)
    df_test.append(temp)

x_test=pd.concat(df_test) # combine test dataset

x_train_1, x_val_1, y_train_1, y_val_1=train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, :-2], train_size=0.7, random_state=0) # Target1 = day 7
x_train_2, x_val_2, y_train_2, y_val_2=train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, :-1], train_size=0.7, random_state=0) # Target2 = day 8

# print(x_train_1.shape) # (36724, 7)
# print(y_train_1.shape) # (36724, 7)
# print(x_val_1.shape) # (15740, 7)
# print(y_val_1.shape) # (15740, 7)
print(x_train_2.shape) # (36724, 7)
print(y_train_2.shape) # (36724, 8)
print(x_val_2.shape) # (15740, 7)
print(y_val_2.shape) # (15740, 8)

# define callbacks
es=EarlyStopping(monitor='val_loss', patience=10, mode='min')
rl=ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_day_3_{epoch:02d}-{val_loss:.4f}.hdf5',
                    save_best_only=True)

# Modeling
input1=Input(shape=(7,))
dense1=Dense(64, activation='relu')(input1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
output1=Dense(1)(dense1)
model1=Model(input1, output1)

model1.compile(loss='mse', optimizer='adam')
model1.fit(x_train_1, y_train_1, validation_split=0.2,
            epochs=100, batch_size=64, callbacks=[es, cp, rl])
y_pred_1=model1.predict(x_test)

input2=Input(shape=(7,))
dense2=Dense(64, activation='relu')(input2)
dense2=Dense(128, activation='relu')(dense2)
dense2=Dense(128, activation='relu')(dense2)
output2=Dense(1)(dense2)
model2=Model(input2, output2)

model2.compile(loss='mse', optimizer='adam')
model2.fit(x_train_2, y_train_2, validation_split=0.2,
            epochs=100, batch_size=64, callbacks=[es, cp, rl])
y_pred_2=model2.predict(x_test)

for i in range(10):
    pred=[]
    pred.append([y_pred_1, y_pred_2])

print(type(pred))

predict=pd.DataFrame(pred)
predict.to_csv('./dacon/sub_mis_1.csv')