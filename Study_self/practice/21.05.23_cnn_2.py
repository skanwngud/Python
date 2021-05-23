import numpy as np
import tensorflow
import datetime

from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.datasets import cifar10

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.saving.save import load_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=23
)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_val = preprocess_input(x_val)

mob = MobileNet(
    input_shape = (32, 32, 3),
    include_top=False)

MobileNet.trainable = True

es = EarlyStopping(patience=20, verbose=1)
rl = ReduceLROnPlateau(patience=10, factor=0.5, verbose=1)
mc = ModelCheckpoint(
    'c:/data/modelcheckpoint/test2.hdf5',
    save_best_only=True,
    verbose=1
)

input = mob.output
x = GlobalAveragePooling2D()(input)
output = Dense(1, activation='softmax')(x)
model = Model(mob.input, output)

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'acc'
)

model.fit(
    x_train, y_train,
    validation_data = (x_val, y_val),
    epochs = 1000,
    batch_size = 32,
    callbacks = [es, rl, mc]
)

model2 = load_model(
    'c:/data/modelcheckpoint/test2.hdf5')

loss, acc = model2.evaluate(x_test, y_test)
y_pred = model2.predict(x_test)

print(f'loss : {loss}')
print(f'acc : {acc}')

print(np.argmax(y_pred[:5], axis = -1))
print(np.argmax(y_test[:5], axis = -1))