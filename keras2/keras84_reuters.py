import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import reuters

(x_train, y_train), (x_test, y_test)=reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0])
print('='*50)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)