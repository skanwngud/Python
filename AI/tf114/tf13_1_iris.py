import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

datasets=load_iris()
x_train = datasets.data # (150, 4)
y_train = datasets.target.reshape(-1, 1) # (150, 1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()

print(y_train.shape) # (150, 3)

x = tf.placeholder(
    tf.float32,
    shape = [None, 4]
)

y = tf.placeholder(
    tf.float32,
    shape = [None, 3]
)

w = tf.Variable(
    tf.random_normal([4, 3]),
    name = 'weights'
)

b = tf.Variable(
    tf.random_normal([1, 3]),
    name = 'bias'
)

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run(
            [loss, hypothesis, train],
            feed_dict={x:x_train, y:y_train}
        )
        if step % 100 == 0:
            print(step, cost_val)
    h, a = sess.run(
        [hypothesis, accuracy],
        feed_dict={x:x_train, y:y_train}
    )
    print(a)

# results
# 0.9822222