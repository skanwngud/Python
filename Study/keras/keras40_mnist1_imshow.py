import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,) <- (10000, 28, 28, 1) 과 같음

print(x_train[0])
print(y_train[0])
print(x_train[0].shape)

plt.imshow(x_train[0], 'gray') # channel 이 1 이므로 흑백
# plt.imshow(x_train[0]) # 'gray'를 하지 않으면 색이 나오지만 정확하지 않음
plt.show()