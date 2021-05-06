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
# 모델을 튜닝을 하며 자동으로 완성해주는 모듈이기 때문에 fit 을 한 다음에 summary 를 찍어야한다


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

model.fit(x_train, y_train,epochs=1, validation_split=0.2, callbacks=[es, rl, mc])
# default - validation_split = 0.2
# callbacks can use

results = model.evaluate(x_test, y_test)

print(results)

# model.summary()

# [0.0784936472773552, 0.9735999703407288]
