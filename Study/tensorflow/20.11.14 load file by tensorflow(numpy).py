import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy=np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=tf.float32)
x_data=xy[:, 0:-1]
y_data=xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

x=tf.placeholder(tf.float32, shape=[None, 3])
y=tf.placeholder(tf.float32, shape=[None, 1])
w=tf.Variable(tf.random_normal([3, 1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis=w*x+b

cost=tf.reduce_mean(tf.square(hypothesis-y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()

sess.run(tf.global_variables_initialize())

for step in range(2001):
    cost_val, hy_val = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if step % 10 == 0:
        print(step, "cost:", cost_val, "\nPrediction:\n", hy_val)

print("Other scroe will be", sess.run(hypothesis, feed_dict={x:[[60, 70, 110], [90, 100, 80]]}))