import tensorflow as tf
import numpy as np
tf.set_random_seed(13)

x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype = np.float32)
y_data = np.array([[0],
                   [1],
                   [1],
                   [0]], dtype = np.float32)

x = tf.placeholder(
    tf.float32,
    shape = [None, 2]
)

y = tf.placeholder(
    tf.float32,
    shape = [None, 1]
)

w1 = tf.Variable(
    tf.random_normal([2, 1]),
    name = 'weights'
)

b1 = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(
    tf.random_normal([1, 1]),
    name = 'weights'
)

b2 = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

hypothesis = tf.sigmoid(tf.matmul(layer1, w2) + b1)

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) + (1-y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        cost_val, hy_val, _ = sess.run(
            [loss, hypothesis, train],
            feed_dict={x:x_data, y:y_data}
        )
        if step % 1000 == 0:
            print(step, cost_val)
    h, a = sess.run(
        [hypothesis, acc],
        feed_dict={x:x_data, y:y_data})
    print(h, a)