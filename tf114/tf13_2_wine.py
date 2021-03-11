import tensorflow as tf
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder

datasets=load_wine()
x_train = datasets.data
y_train = datasets.target.reshape(-1, 1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()

print(x_train.shape) # (178, 13)
print(y_train.shape) # (178, 3)

x = tf.placeholder(
    tf.float32,
    shape = [None, 13]
)

y = tf.placeholder(
    tf.float32,
    shape = [None, 3]
)

w = tf.Variable(
    tf.zeros([13, 3]),
    name = 'weights'
)

b = tf.Variable(
    tf.zeros([1, 3]),
    name = 'bias'
)

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        cost_val, hy_val, _ = sess.run(
            [loss, hypothesis, train],
            feed_dict={x:x_train, y:y_train}
        )
        if step % 1000 == 0:
            print(step, cost_val)
    h, a = sess.run(
        [hypothesis, accuracy],
        feed_dict={x:x_train, y:y_train}
    )
    print(np.argmax(h, axis = -1)[:5], a)