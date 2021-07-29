import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import reuters # 로이터 뉴스 기사
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, \
    BatchNormalization, Activation, Input, Dropout, Conv1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test)=reuters.load_data(
    num_words=10000, test_split=0.2
)

# print(x_train[0]) # list
# print(y_train[0]) # 3
# print(len(x_train[0]), len(x_train[11])) # 87, 59 -> 문장의 길이가 다름

# print('='*50)

# print(x_train.shape, x_test.shape) # (8982, ) (2246, )
# print(y_train.shape, y_test.shape) # (8982, ) (2246, )

# print('뉴스기사 최대 길이 : ', max(len(l) for l in x_train)) # for 문을 돌면서 가장 큰 값을 반환 / 2376
# print('뉴스기사 평균 길이 : ', sum(map(len, x_train))/ len(x_train)) # map 알아볼 것  / 145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# y 분포
unique_elemnets, counts_elemnets=np.unique(
    y_train,
    return_counts=True
)

# print('y 분포 : ', dict(zip(unique_elemnets, counts_elemnets))) # dict -> dictionary 형태, zip -> a,b 를 각 행끼리 a : b 의 형태로 짝지어줌
# print('='*50)

# plt.hist(y_train, bins=46)
# plt.show()

# x 의 단어들 분포
word_to_index=reuters.get_word_index() # reuters 에 있는 단어들을 가져올 수 있다
# print(word_to_index)
# print(type(word_to_index)) # <class 'dict'>
# print('='*50)

# Key 와 Value 를 바꿔줌
index_to_word=dict()

for key, value in word_to_index.items():
    index_to_word[value]=key

# Key 와 Value 를 바꿔준 후 출력
# print(index_to_word)
# print(index_to_word[1]) # 첫 번재로 많이 쓰인 단어 'the'
# print(index_to_word[30979]) # 제일 덜 쓰인 단어 'northerly'
# print(len(index_to_word))

# x_train[0]
# print(x_train[0])
# print(' '.join([index_to_word[index] for index in x_train[0]]))

# y 카테고리 갯수 출력
category=np.max(y_train) + 1 # y 의 최대값 45
print('y 카테고리 : ', category) # 46

# y 의 유니크한 값 출력
y_bunpo=np.unique(y_train)
print('y 유니크 : ', y_bunpo)

pad_x=pad_sequences(
    x_train,
    padding='pre',
    maxlen=100
)

pad_x_test=pad_sequences(
    x_test,
    padding='pre',
    maxlen=100
)

print(pad_x.shape) # (8982, 100)
# print(pad_x[0])

# pad_x=pad_x.reshape(-1, 2376, 1)

# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)

es=EarlyStopping(
    patience=70,
    verbose=1
)

mc=ModelCheckpoint(
    'c:/data/modelcheckpoint/keras84.hdf5',
    save_best_only=True,
    verbose=1
)

rl=ReduceLROnPlateau(
    patience=30,
    factor=0.5,
    verbose=1
)

model=Sequential()
# model.add(Embedding(
#     input_dim=10000,
#     output_dim=64,
#     input_length=100))
model.add(Embedding(
    10000, 64
))
model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Conv1D(256, 2))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(46, activation='softmax'))

# model.summary()

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics='acc'
# )

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='acc'
)

model.fit(
    pad_x, y_train,
    epochs=200,
    validation_split=0.2,
    batch_size=32,
    callbacks=[es, rl]
)

loss=model.evaluate(
    pad_x_test, y_test
)

print('loss : ', loss[0])
print('acc : ', loss[1])

# results
# loss :  1.933007836341858
# acc :  0.6936776638031006

# loss :  3.7020275592803955
# acc :  0.6941229104995728