import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 784).astype('float32')/255
x_test = x_test.reshape(-1, 784)/255.

# print(x_train.shape) # (60000, 784)
# print(x_test.shape) # (10000, 784)
# print(y_train.shape) # (60000, 10)
# print(y_test.shape) # (10000, 10)

x = tf.placeholder(
    tf.float32,
    shape = [None, 784]
)

y = tf.placeholder(
    'float32',
    [None, 10]
)

w = tf.Variable(
    tf.random_normal([784, 10]),
    name = 'weight'
)

b = tf.Variable(
    tf.random_normal([10]),
    name = 'bias'
)

# 2. modeling
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# 3. compile, fitting (multiple classification)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _, cost_val = sess.run(
            [train, loss],
            feed_dict={x:x_train, y:y_train}
        )
        