import numpy as np
import tensorflow

from keras.preprocessing.text import Tokenizer

docs=[
    '너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘 생기긴 했어요',
]

# 긍정 1, 부정 0
labels=np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token=Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x=token.texts_to_sequences(docs)
print(x)

# 패딩
from keras.preprocessing.sequence import pad_sequences
pad_x=pad_sequences(x, padding='pre', maxlen=5) # 'pre' - 앞쪽부터, 'post' - 뒤쪽부터

print(pad_x)
print(pad_x.shape) # (13, 5) : 중복 제외 13개의 단어, 가장 긴 문장에 들어간 단어의 갯수 5개 / maxlen = 4 -> (13, 4)

print(np.unique(pad_x)) # 객체별로 수치가 나옴

print(len(np.unique(pad_x))) # 28 (token 은 0 제외, np.unique 는 0 을 포함해서 센다)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model=Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# model.add(Embedding(28, 11)) # ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.
# model.add(LSTM(32))
model.add(Conv1D(32, 2))
model.add(Flatten()) # Flatten 을 해주지 않아도 되지만 성능이 떨어지게 됨 0.8462 -> 1.0
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='acc'
)

model.fit(
    pad_x, labels,
    epochs=100
)

acc=model.evaluate(
    pad_x, labels
)[1] # [0] = loss, [1] = metrics

print('acc : ', acc)
