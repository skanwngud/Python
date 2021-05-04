# best model save

import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier( 
    overwrite=True,       
    max_trials=1,
    loss='mse',
    metrics=['acc']
)

# model.summary()


es = EarlyStopping(
    verbose=1, patience=10
)
rl = ReduceLROnPlateau(
    verbose=1, patience=5
)
mc = ModelCheckpoint(
    './temp/',
    verbose=1, save_best_only=True
)

model.fit(
    x_train, y_train,
    epochs=1,
    validation_split=0.2, 
    callbacks=[es, rl, mc]
)
# default - validation_split = 0.2
# callbacks can use

results = model.evaluate(x_test, y_test)

print(results)

# model.summary()

model2=model.export_model()
model2.save('c:/data/h5/autokeras_1.h5')

best_model = model.tuner.get_best_model()
best_model.save('c:/data/h5/best_ak.h5')

# [0.06337708234786987, 0.9793000221252441]