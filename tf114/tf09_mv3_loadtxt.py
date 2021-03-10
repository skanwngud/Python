# 73, 80, 75, 152
# 

import tensorflow as tf
import numpy as np
tf.set_random_seed(23)

dataset = np.loadtxt('c:/data/csv/data-01-test-score.csv', delimiter=',')
x_data = dataset[5:, :-1]
y_data = dataset[5:, -1]
y_data = y_data.reshape(-1, 1)

x_predict = dataset[:5, :-1]

print(x_data.shape) # (25, 3)
print(y_data.shape) # (25, )

# print(x_data[:5])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(
    tf.random_normal([3, 1]),
    name = 'weights'
)
b = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={x:x_data, y:y_data}
    )
    if step % 1000 == 0 :
        print('step : ', step, '\ncost : ', cost_val, '\nhypothesis : \n', hy_val)
print(sess.run(hypothesis, feed_dict={x:x_predict}))

# results
# step :  10000 
# cost :  6.131322 
# hypothesis :
#  [[152.63782]
#  [185.00073]
#  [181.44722]
#  [198.87747]
#  [140.292  ]]

# [[153.20647]
#  [184.81697]
#  [181.65193]
#  [199.07674]
#  [140.09544]]