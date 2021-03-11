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
    tf.random_normal([2, 128]), # 히든레이어의 노드수를 조절하기 위해선 w 의 열 갯수를 조정하면 된다
    name = 'weights'
)

b1 = tf.Variable(
    tf.random_normal([128]), # w 가 (2, 10) 이므로 bias 값은 10 이다
    name = 'bias'
)

layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(10, input_dim=2, activation = 'sigmoid'))

w2 = tf.Variable(
    tf.random_normal([128, 64]),
    name = 'weights2'
)

b2 = tf.Variable(
    tf.random_normal([64]),
    name = 'bias2'
)

layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
# model.add(Dense(7, activation = 'sigmoid'))

w3 = tf.Variable(
    tf.random_normal([64, 1]), # output 값을 맞춰야한다
    name = 'weights3'
)

b3 = tf.Variable(
    tf.random_normal([1]),
    name = 'bias3'
)

hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)
# model.add(Dense(1, activation = 'sigmoid'))

loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) + (1-y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        cost_val, hy_val, _ = sess.run(
            [loss, hypothesis, train],
            feed_dict={x:x_data, y:y_data}
        )
    h, a = sess.run(
        [hypothesis, acc],
        feed_dict={x:x_data, y:y_data})
    print(a)