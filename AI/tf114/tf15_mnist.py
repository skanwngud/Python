import tensorflow as tf
import numpy as np

from keras.datasets import mnist

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
tf.set_random_seed(12)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000, )
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000, )

x_train = x_train.reshape(-1, 784)/255.
x_test = x_test.reshape(-1, 784)/255.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

print(y_train.shape) # (60000, 10)
print(y_train[0])

####

x = tf.placeholder(
    tf.float32,
    shape = [None, 784]
)

y = tf.placeholder(
    tf.float32,
    shape = [None, 10]
)

w = tf.Variable(
    tf.random.normal([784, 512], stddev = 0.1), # stddev : default 1.0, random_normal 의 범위를 정해준다.
    name = 'weights'                            # 이 경우에는 1.0 으로 주게 되면 nan 이 나와서 범위를 줄여주었다.
)

b = tf.Variable(
    tf.random.normal([512], stddev = 0.1),
    name = 'bias'
)

w2 = tf.Variable(
    tf.random.normal([512, 256], stddev = 0.1),
    name = 'weights2'
)

b2 = tf.Variable(
    tf.random.normal([256], stddev = 0.1),
    name = 'bias2'
)

w3 = tf.Variable(
    tf.random.normal([256, 64], stddev = 0.1),
    name = 'weights3'
)

b3 = tf.Variable(
    tf.random.normal([64], stddev = 0.1),
    name = 'bias3'
)

w4 = tf.Variable(
    tf.random.normal([64, 10], stddev = 0.1),
    name = 'weights4'
)

b4 = tf.Variable(
    tf.random.normal([10], stddev = 0.1),
    name = 'bias4'
)

layer1 = tf.nn.relu(tf.matmul(x, w) + b)
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# predict = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32))

predict = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20):
        cost_val, hy_val, _ = sess.run(
            [loss, hypothesis, train],
            feed_dict={x:x_train, y:y_train}
        )
        print(step, cost_val)
    y_pred = sess.run(
        hypothesis,
        feed_dict={x:x_test}
    )
    y_pred = np.argmax(y_pred, axis = -1)
    y_test = np.argmax(y_test, axis = -1)
    print('acc : ', accuracy_score(y_test, y_pred))
    print('y_test : ', y_test[:5])
    print('hy_val : ', y_pred[:5])

# results
# acc :  0.9086
# y_test :  [7 2 1 0 4]
# hy_val :  [7 2 1 0 4]