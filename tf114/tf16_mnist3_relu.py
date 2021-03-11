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

# 2. modeling
x = tf.placeholder(
    tf.float32,
    shape = [None, 784]
)

y = tf.placeholder(
    'float32',
    [None, 10]
)

tf.initializers.glorot_normal()

w = tf.Variable(
    tf.random_normal([784, 100], stddev=0.1),
    name = 'weight'
)

b = tf.Variable(
    tf.random_normal([100], stddev=0.1),
    name = 'bias'
)

w2 = tf.Variable(
    tf.random_normal([100, 50], stddev=0.1),
    name = 'weight2'
)

b2 = tf.Variable(
    tf.random_normal([50], stddev=0.1),
    name = 'bias2'
)

w3 = tf.Variable(
    tf.random_normal([50, 10], stddev=0.1),
    name = 'weight3'
)

b3 = tf.Variable(
    tf.random_normal([10], stddev=0.1),
    name = 'bias3'
)

# layer1 = tf.nn.softmax(tf.matmul(x, w) + b) # softmax 는 중간에 끼면 성능이 제대로 안 나오는 경우가 많다
# layer1 = tf.nn.relu(tf.matmul(x, w) + b)
# layer1 = tf.nn.selu(tf.matmul(x, w) + b)
layer1 = tf.nn.elu(tf.matmul(x, w) + b) # relu, selu, elu 가능
layer1 = tf.nn.dropout(layer1, keep_prob=0.3) # model.add(Dropout(0.3))

layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)

hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

# 3. compile, fitting (multiple classification)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _, cost_val = sess.run(
            [train, loss],
            feed_dict={x:x_train, y:y_train}
        )
        if step % 200 == 0:
            print(step, cost_val)

            # 2000 2.2656002