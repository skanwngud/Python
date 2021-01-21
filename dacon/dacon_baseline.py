# import library
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

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

# define callbacks
es=EarlyStopping(monitor='val_loss', patience=10, mode='min')
rl=ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_day_3_{epoch:02d}-{val_loss:.4f}.hdf5',
                    save_best_only=True)

print(x_train_1.shape) # (326724, 7)
print(y_train_1.shape) # (326724, 7)
print(x_test.shape) # (3888, 7)

# Modeling
quantile_list=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def quantile_loss_dacon(q, y_true, y_pred):
    err=(y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

def models(quantile_list, x_train, y_train, x_val, y_val, x_test):
    model=Sequential()
    model.add(Dense(64, activation='relu', input_shape=(7,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2*48*2))
    # model.add(Reshape(2, 48, 2))
    model.add(Dense(1))

    model.summary()

    model.compile(loss=lambda y_true, y_pred:quantile_loss_dacon(quantile_list, y_true, y_pred), optimizer='adam')
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=100, batch_size=64, callbacks=[es, rl])

    model.evaluate(x_val, y_val)
    pred=model.predict(x_test)
    pred=pred.reshape(-1*1)
    pred=pd.Series(pred.round(2))
    print(pred.shape)
    print(type(pred))
    return model, pred

def train_data(x_train, y_train, x_val, y_val, x_test):
    dacon_models=list()
    dacon_actual_pred=pd.DataFrame()
    print(type(dacon_actual_pred))

    for q in quantile_list:
        pred, model=models(q, x_train, y_train, x_val, y_val, x_test)
        dacon_models.append(model)
        dacon_actual_pred=pd.concat([dacon_actual_pred, pred], axis=1)
    
    dacon_actual_pred.columns=quantile_list
    return dacon_models, dacon_actual_pred

models_1, results_1=train_data(x_train_1, y_train_1, x_val_1, y_val_1, x_test)
results_1.sort_index()[:48]