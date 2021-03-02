import tensorflow

from keras.applications import VGG16 # 레이어가 16개인 모델
from keras.layers import Dense, Flatten
from keras.models import Sequential

vgg16=VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

# print(model.weights)

vgg16.trainable=False # True 를 하게 되면 최적의 weights 가 틀어지게 됨

vgg16.summary()

print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.summary()

print('그냥 가중치의 수 : ', len(model.weights)) # 26 -> 32
print('동결하기(trainable = Falser) 전 훈련 되는 가중치의 수 : ', len(model.trainable_weights)) # 0 -> 6 - Sequential model 의 weights, bias

import pandas as pd
pd.set_option('max_colwidth', -1) # set_option 찾아보기
layers=[(layer, layer.name, layer.trainable) for layer in model.layers]
aaa=pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)
'''
                                                                            Layer Type Layer Name  Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x000001590619ADC0>  vgg16      False -> 훈련하지 않음
1  <tensorflow.python.keras.layers.core.Flatten object at 0x00000159064789A0>           flatten    True -> 레이어를 하나씩 지정해줘서 훈련 시키지 않을 수도 있지만
2  <tensorflow.python.keras.layers.core.Dense object at 0x0000015906497880>             dense      True -> 굳이 그럴 필요가 없음 (안 넣으면 되기 때문)
3  <tensorflow.python.keras.layers.core.Dense object at 0x00000159064D5250>             dense_1    True
4  <tensorflow.python.keras.layers.core.Dense object at 0x00000159064DED60>             dense_2    True
'''