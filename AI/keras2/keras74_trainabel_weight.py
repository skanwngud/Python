import numpy as np
import tensorflow

from keras.models import Sequential
from keras.layers import Dense

#1. data
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])

#2. model
model=Sequential()
model.add(Dense(4, input_shape=(1,)))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# print(model.weights)

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[ 0.60442924,  0.6816293 , -0.01906002,  0.26398814]], -> 다음 레이어로 전달해주는 weight 값 : 4개 (1, 4)
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.] -> 바이어스 값, dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.8082031 , -0.4557837 ,  0.61142457],
       [-0.78954744,  0.13542545,  0.78517497], -> 12개 (4, 3)
       [-0.26728064, -0.15570861,  0.7943618 ],
       [-0.33097148,  0.01039165,  0.9134481 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, 
numpy=
array([[ 1.0097449 ,  0.80027544], -> 6개 (3, 2)
       [-0.85779935,  0.77730095],
       [-0.60865694, -0.9968494 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.59734696], -> 2개 (2, 1)
       [-0.6324047 ]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

# print(model.trainable_weights) # 위 모델에서 훈련 시키는 가중치 값 (전이학습 사용시 non-trainable_weights 를 사용)

''' -> model.weights 와 동일 : 전부 훈련을 시키기 때문
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=array([[-0.7565856,  0.9506748,  0.2583859,  0.5298115]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[-0.88290304, -0.20222694,  0.738132  ],
       [ 0.03413767, -0.584662  ,  0.7224995 ],
       [-0.16715336,  0.6482624 , -0.87578136],
       [ 0.3566023 , -0.6340638 ,  0.66827893]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, 
numpy=
array([[ 0.85795736, -0.6903852 ],
       [ 0.61680925, -0.96108294],
       [ 0.82254374, -1.0401884 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.47017908],
       [ 0.937271  ]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print(len(model.weights)) # 8 -> 각 layer 당 갖고있는 weight 와 bias 갯수 (4 layer 에 하나씩 weight, bias (2))
print(len(model.trainable_weights)) # 8