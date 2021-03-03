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
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12,
# '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17,
# '생각보다 : 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '규현이가': 25, '생기긴': 26, '했어요': 27}

x=token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
# 크기가 다 다르기 때문에 각자의 크기를 맞춰줘야한다
# LSTM 의 경우 뒤로 갈수록 훈련의 성과가 커지기 때문에 길이를 맞출 땐 보통 앞 쪽에 0을 채운다

# 패딩
from keras.preprocessing.sequence import pad_sequences
pad_x=pad_sequences(x, padding='pre', maxlen=5) # 'pre' - 앞쪽부터, 'post' - 뒤쪽부터
# 자동으로 가장 긴 문장 기준으로 크기를 맞춰준다

print(pad_x)
print(pad_x.shape) # (13, 5) : 중복 제외 13개의 단어, 가장 긴 문장에 들어간 단어의 갯수 5개 / maxlen = 4 -> (13, 4)
# (13, 5) 중 앞 뒤로 일정부분을 날리기 위해선 maxlen 사용

print(np.unique(pad_x)) # 객체별로 수치가 나옴
# [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

print(len(np.unique(pad_x))) # 28 (token 은 0 제외, np.unique 는 0 을 포함해서 센다)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model=Sequential()
# model.add(Embedding(input_dim=28, output_dim=64, input_length=5)) # 일반적인 레이어와 다르게 인풋이 먼저 들어감
# input_dim(word_size) = x에 들어가는 종류가 총 몇 가지인지 (이 경우엔 28개)
# input_length = pad_x.shape 의 경우 (13, 5) 이기 때문에 5 를 준다 column 수
# 임베딩 레이어를 들어갈 때 인풋레이어가 단어갯수보다 적으면 돌아가지 않고 늘리면 연산량은 늘어나지만 훈련 자체는 됨
model.add(Embedding(28, 64)) # 파라미터를 명시하지 않은 경우 자동적으로 인식함
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary() # 총 단어의 갯수 28 * 아웃풋 노드 64 = 28*64==1792

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='acc'
)

model.fit(
    pad_x, labels,
    epochs=30
)

acc=model.evaluate(
    pad_x, labels
)[1] # [0] = loss, [1] = metrics

print('acc : ', acc)

"""
벡터화
pad_sequence 시 수치를 원핫인코딩을 실시하게 되면 문장이 길어질 경우에 수치가 엄청나게 길어지게 되지만,
벡터화를 시키게 되면 x,y 의 좌표상에 찍어주게 되므로 크기를 줄일 수 있다
(eg. 0 = 1 0 0 0 을 0 = 0.22 로 바꿔줌)
"""

# results

# LSTM
# acc :  0.9230769276618958

# Conv1D
# acc :  0.8461538553237915

# Dense
# acc :  0.7846154570579529