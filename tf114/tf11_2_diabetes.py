import tensorflow as tf

from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x_train = datasets.data # (442, 10)
y_train = datasets.target.reshape(-1, 1) # (442, 1)

x = tf.placeholder(
    tf.float32,
    shape=[None, 10]
)

y = tf.placeholder(
    tf.float32,
    shape=[None, 1]
)

w = tf.Variable(
    tf.random_normal([10, 1],
    name = 'weights')
)

b = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
train = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict = {x:x_train, y:y_train}
        )
        if step % 1000 == 0:
            print(step, cost_val)

# results
# r2 스코어로 불러올 것
# 10000 2877.519