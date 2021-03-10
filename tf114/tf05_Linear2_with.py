import tensorflow as tf

tf.set_random_seed(23)

x_train = [1, 2, 3]
y_train = [3, 5, 7]

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
    for step in range(1001): # 0~1000 번 == epochs
        sess.run(train)
        if step <= 3: # step 이 20번일 때마다 해당 내용을 반환함
            print(step, sess.run(cost), sess.run(W), sess.run(b))
# with 문 안에 들어가게 되면 sess.close() 를 쓸 필요 없음