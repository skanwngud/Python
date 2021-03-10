import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 51, 65],
        [92, 98, 11],
        [89, 31, 33],
        [99, 33, 100],
        [17, 66, 79]]
y_data = [[152],
        [185],
        [180],
        [205],
        [142]]

x = tf.placeholder(
    tf.float32,
    shape=[None, 3] #  x_data 의 shape. (5, 3) 이므로 행무시
)

y = tf.placeholder(
    tf.float32,
    shape=[None, 1]
)

w = tf.Variable(
    tf.random_normal([3, 1]),
    name = 'weight'
) # 행렬의 곱

b = tf.Variable(
    tf.random_normal([1]),
    name = 'bias'
)

# W * x 의 형태와 y 의 형태가 같아야함

# hypothesis = w * x + b
hypothesis = tf.matmul(x, w) + b # 행렬의 곱

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    _, cost_val, hy_val = sess.run(
        [train, cost, hypothesis],
        feed_dict={x:x_data, y:y_data}
    )
    if step % 1000 == 0:
        print(step, cost_val, hy_val)

sess.close()