# placeholder 사용

import tensorflow as tf

tf.set_random_seed(23)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(
    tf.random_normal([1]), name='weight'
)
b = tf.Variable(
    tf.random_normal([1]), name='bias'
) # random_normal : random한 값 하나를 normalization 하여 집어넣음

# 모델, 로스, 최적화함수 선언
hypothesis = x_train * W + b # linear 모델
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss mse

# 훈련
train = tf.train.GradientDescentOptimizer(
    learning_rate=0.01
).minimize(cost) # optimizer 와 train 을 같이 묶음 

# for step in range(1001): # 0~1000 번 == epochs
#     sess.run(train)
#     if step <= 3: # step 이 20번일 때마다 해당 내용을 반환함
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

# sess.close()



with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(2001): # 0~1000 번 == epochs
        _, cost_val, W_val, b_val=sess.run(
            [train, cost, W, b],
            feed_dict={x_train:[1, 2, 3], y_train : [3, 5, 7]}
        ) # 최초의 placeholder 는 한 번만 들어감 // sess.run 을 하게 되면서 반환값이 출력
        if step % 20 == 0: # step 이 20번일 때마다 해당 내용을 반환함
            print(step, cost_val, W_val, b_val)
    print(sess.run(hypothesis, feed_dict={x_train:[4], y_train : [3, 5, 7]}))
    print(sess.run(hypothesis, feed_dict={x_train:[5, 6], y_train : [3, 5, 7]}))
    print(sess.run(hypothesis, feed_dict={x_train:[6, 7, 8], y_train : [3, 5, 7]}))
# with 문 안에 들어가게 되면 sess.close() 를 쓸 필요 없음

# [4], [5,6], [7,8,9] predict 할 것
# [9.002323]
# [11.003669 13.005014]
# [13.005014 15.00636  17.007706]
