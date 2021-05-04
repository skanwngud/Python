import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = ak.ImageClassifier( # shape 에 따라서 각 모델들이 존재 (지금은 이미지이므로 ImageClassifier)
    # overwrite=True,       # y 를 one_hot_encoding 하지 않음 (쓰게 되면 output 값의 shape 가 달라짐)
    max_trials=1            # ImageClassifier 의 경우 to_categorical 가능
)

model.fit(x_train, y_train, epochs=1)

results = model.evaluate(x_test, y_test)

print(results)

# [0.06339078396558762, 0.9789000153541565]