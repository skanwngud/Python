# 실습
# feature_importances 를 y 에도 사용

import numpy as np
import pandas as pd
import tensorflow
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, LSTM, Flatten,\
    BatchNormalization, Activation, Input, concatenate
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV


es=EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    verbose=1
)

rl=ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=30,
    verbose=1
)

mc=ModelCheckpoint(
    'c:/data/modelcheckpoint/keras86.hdf5',
    save_best_only=True,
    verbose=1
)

kf=KFold(
    n_splits=15,
    shuffle=True,
    random_state=23
)

df=pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')

# print(df.info()) # 4898, 12
# print(df.head())

x=df.iloc[:, :-1].values # (4898, 11)
y=df['quality'].values # (4898)


scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

# x=x.reshape(-1, 11, 1)
y=y.reshape(-1, 10, 1)

one=OneHotEncoder()
one.fit(y)
y=one.transform(y).toarray()

# y=to_categorical(y)

# print(x.shape)
# print(x[:5])
# print(y.shape) # (10, )

# x_train, x_test, y_train, y_test=train_test_split(
#     x,y,
#     train_size=0.8,
#     random_state=23
# )


for train_index, test_index in kf.split(x,y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

input=Input(shape=(11, ))
lstm=Dense(256)(input)
bat=BatchNormalization()(lstm)
act=Activation('relu')(bat)
dense=Dense(128)(act)
bat=BatchNormalization()(dense)
act=Activation('relu')(bat)
dense=Dense(256)(act)
bat=BatchNormalization()(dense)
act=Activation('relu')(bat)
x=Dense(256)(act)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=Dense(128)(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=Dense(64)(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)


output=Dense(10, activation='softmax')(x)
model=Model(input, output)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=16,
    callbacks=[es, rl, mc]
)

model.load_weights(
    'c:/data/modelcheckpoint/keras86.hdf5'
)

loss=model.evaluate(
    x_test, y_test
)

print('Loss : ', loss[0])
print('Acc : ', loss[1])

# results (Robust)
# Loss :  1.5185446739196777
# Acc :  0.5836734771728516

# results (Dense)
# Loss :  0.99104243516922
# Acc :  0.6165643930435181