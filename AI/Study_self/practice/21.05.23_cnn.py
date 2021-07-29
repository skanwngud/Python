import numpy as np
import tensorflow
import datetime

from keras.layers import Conv2D, BatchNormalization, Activation,\
    Flatten, Dense, Input
from keras.models import Sequential, Model, load_model
from keras.datasets import cifar10, fashion_mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Dropout

str_time = datetime.datetime.now()

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=23
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

print(x_train.shape) # (40000, 32, 32, 3)
print(y_train.shape) # (40000, 10)

# define functions
def cnn_layer(filter, kernel, stride, input):
    x = Conv2D(filter, kernel, stride, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

es = EarlyStopping(patience=20, verbose=1)
rl = ReduceLROnPlateau(patience=10, verbose=1, factor=0.5)
mc = ModelCheckpoint('c:/data/modelcheckpoint/test.hdf5', verbose=1, save_best_only=True)

# model
input = Input(shape = (x_train.shape[1], x_train.shape[2], 1))
layer = cnn_layer(32, 2, 2, input)
layer = cnn_layer(32, 2, 2, layer)
layer = cnn_layer(32, 2, 2, layer)
layer = Dropout(0.2)(layer)
layer = cnn_layer(64, 2, 2, layer)
layer = cnn_layer(64, 2, 2, layer)
layer = cnn_layer(64, 2, 2, layer)
layer = Dropout(0.2)(layer)
layer = Flatten()(layer)
output = Dense(10, activation = 'softmax')(layer)
model = Model(input, output)

# compile, fit
model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics = 'acc'
)

model.fit(
    x_train, y_train,
    validation_data = (x_val, y_val),
    epochs = 5,
    batch_size = 32,
    callbacks = [es, rl, mc]
)

model2 = load_model(
    'c:/data/modelcheckpoint/test.hdf5'
)

# evaluate, predict
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print(f'loss : {loss[0]}')
print(f'acc : {loss[1]}')

print(np.argmax(y_pred[:5], axis = -1))
print(np.argmax(y_test[:5], axis = -1))
print(datetime.datetime.now() - str_time)