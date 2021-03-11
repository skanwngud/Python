import tensorflow as tf
import numpy as np
tf.set_random_seed(11)

# 데이터
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1], # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0], # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0], # 0
          [1, 0, 0]] # OneHotEncoding

# 변수 선언
x = tf.placeholder(
    'float',
    [None, 4] # 행이 늘어남 : 데이터가 늘어남 / 열이 늘어남 : 특성이 늘어남
)

y = tf.placeholder(
    'float',
    [None, 3] # OneHotEncoding 을 했으므로 y = (8, 1) 에서 (8, 3) 이 된다
)

w = tf.Variable(
    tf.random_normal([4, 3]), # (N, 4) * (4, 3) = (N, 3)
    name = 'weights'
)

b = tf.Variable(
    tf.random_normal([1, 3]), # w 의 값이 (N, 3) 이 되므로 행렬의 합을 위해 shape 를 맞춰줘야하므로 (1, 3) 이 된다
    name = 'bias'
)

# 모델링
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # shape = (N, 3) / 위 경우엔 (8, 3)
# loss = tf.reduce_mean(tf.square(hypothesis - y)) -> mse
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1 - hypothesis)) -> binsary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        _, cost_val = sess.run(
            [optimizer, loss],
            feed_dict={x:x_data, y:y_data}
        )
        if step % 200 == 0:
            print(step, cost_val)
# predict
    a = sess.run(
        hypothesis,
        feed_dict={x:[[1, 11, 7, 9]]})

    print(a, sess.run(
        tf.argmax(a, 1) # 가장 높은 값에 1 을 반환시킴
    ))

    # print(np.argmax(a, axis=-1))