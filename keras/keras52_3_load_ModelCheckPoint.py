random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model1=load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5')
loss2=model1.evaluate(x_test, y_test)
y_pred2=model1.predict(x_test)

print('로드체크포인트_loss : ', loss2[0])
print('로드체크포인트_acc : ', loss2[1])


# results
# 가중치_loss :  0.044517237693071365
# 가중치_acc :  0.9904000163078308
# 로드모델_loss :  0.044517237693071365
# 로드모델_acc :  0.9904000163078308
# 로드체크포인트_loss :  0.037594024091959 - 1
# 로드체크포인트_acc :  0.9908000230789185 - 1
# 로드체크포인트_loss :  0.037594024091959 - 2
# 로드체크포인트_acc :  0.9908000230789185 - 2

### 가중치, 로드모델 < 로드체크포인트 - 가중치, 로드모델의 경우 얼리스타핑 이후 patience 가 지난 시점의 결과값이라 로드체크포인트보다 좋지 않다