# 이진분류
# acc 스코어

import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
x_train = datasets.data
y_train = datasets.target.reshape(-1, 1)

print(x_train.shape) # (569, 30)
print(y_train.shape) # (569, 1)

x = tf.placeholder(
    tf.float32,
    shape = [None, 30]
)

y = tf.placeholder(
    tf.float32,
    shape = [None, 1]
)

w = tf.Variable(
    tf.random_normal([30, 1]),
    name = 'weights'
)

b = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1001):
        cost_val, _ = sess.run(
            [cost, train],
            feed_dict = {x:x_train, y:y_train}
        )
        if step % 100 == 0:
            print(step, cost_val)

    # h, c, a = sess.run(
    #     [hypothesis, predict, accuracy],
    #     feed_dict = {x:x_train, y:y_train}
    # )
    # print(h, c, a)
