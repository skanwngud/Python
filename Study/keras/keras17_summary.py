import numpy as np
import tensorflow as tf

x = np.array([1,2,3])
y = np.array([1,2,3])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(8, name='aaaa'))
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 10          - (None, 5)==(5,) 형태의 데이터
# _________________________________________________________________
# dense_1 (Dense)              (None, 3)                 18          - Pram#(Parameter Number)==연산의 갯수, ((인풋 레이어 노드 + 1(바이어스))) * 아웃풋 레이어 노드)
# _________________________________________________________________
# dense_2 (Dense)              (None, 4)                 16          - Dense_2(Dense)==임의로 레이어의 이름을 지어줌
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 49
# Trainable params: 49
# Non-trainable params: 0
# _________________________________________________________________

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 7)                 14
# _________________________________________________________________
# dense_1 (Dense)              (None, 6)                 48
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 21
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 87
# Trainable params: 87
# Non-trainable params: 0
# _________________________________________________________________