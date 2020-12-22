import tensorflow as tf
import numpy as np
from builtins import print

x_data=np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data=np.array([[0],[1],[1],[0]], dtype=np.float32)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
w=tf.Variable(tf.random_normal([2,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis=tf.sigmoid(tf.matmul(x,w)+b)

cost=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}), sess.run(w))

    h,c,a = sess.run([hypothesis, prediction, accuracy], feed_dict={x:x_data, y:y_data})
    print=("\nprediction:", c, "\naccuracy:", a, "\nhypothesis:", h)