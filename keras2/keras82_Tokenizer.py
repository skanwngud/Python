import tensorflow

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

text='나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

# 어절별로 자르기 (띄어쓰기 기준)
token=Tokenizer()
token.fit_on_texts([text])

print(token.word_index) # {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}
# word_index = 단어의 빈도수가 높은 순으로 정렬 / 빈도수가 동일할 땐 순서대로 번호를 부여함

x=token.texts_to_sequences([text])

print(x) # [[3, 1, 1, 4, 5, 1, 2, 2, 6]]
# 정의 된 단어의 번호로 나옴

# OneHotEncoding / to_categorical
word_size=len(token.word_index)

print(word_size) # 6 (반복 되는 어절 제외 총 6개)

x=to_categorical(x)

print(x)
print(x.shape) # (1, 9, 7)