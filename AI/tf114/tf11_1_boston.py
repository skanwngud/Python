import tensorflow as tf

from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

datasets = load_boston()
x_train = datasets.data # (506, 13)
y_train = datasets.target.reshape(-1, 1) # (13, 1)

x = tf.placeholder(
    tf.float32,
    shape = [None, 13])

y = tf.placeholder(
    tf.float32,
    shape = [None, 1])

w = tf.Variable(
    tf.random_normal([13, 1]),
    name = 'weights'
)

b = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict={x:x_train, y:y_train}
        )
        if step % 1000 == 0:
            print(step, cost_val)

# results
# 20000 55.15824