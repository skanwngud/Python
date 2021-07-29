# dataset for embedding
# embedding 으로 모델 만들 것

import tensorflow
import matplotlib.pyplot as plt

from keras.datasets import reuters, imdb
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, \
    LSTM, Conv1D, BatchNormalization, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test)=imdb.load_data(
    num_words=10000
) # 이진분류

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

pad_x=pad_sequences(
    x_train,
    padding='pre',
    maxlen=500
)

pad_x_test=pad_sequences(
    x_test,
    padding='pre',
    maxlen=500
)

# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)


print(pad_x.shape) # (25000, 100)
print(y_train.shape) # (25000, 2)

es=EarlyStopping(
    patience=5,
    verbose=1
)

# rl=ReduceLROnPlateau(
#     patience=20,
#     verbose=1,
#     factor=0.5
# )

model=Sequential()
model.add(Embedding(10000, 500))
model.add(BatchNormalization())
model.add(Conv1D(256, 2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(LSTM(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics='acc'
)

model.fit(
    pad_x, y_train,
    validation_split=0.2,
    batch_size=64,
    epochs=10,
    callbacks=[es]
)

loss=model.evaluate(
    pad_x_test, y_test
)

print('loss : ', loss[0])
print('acc : ', loss[1])

# results
# loss :  0.8326995968818665
# acc :  0.8356000185012817

# loss :  0.6493784785270691
# acc :  0.8680400252342224