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

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # variable 을 선언했기 때문에 초기화

print(sess.run(W), sess.run(b)) # [0.13260844] [0.32654932]

# 모델, 로스, 최적화함수 선언
hypothesis = x_train * W + b # linear 모델
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss mse
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01
) # optimizer SGD

# 훈련
train = optimizer.minimize(cost) # optimizer 에 cost 를 넣고 최소화 시킴 (훈련)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001): # 0~1000 번 == epochs
    sess.run(train)
    if step <= 3: # step 이 20번일 때마다 해당 내용을 반환함
        print(step, sess.run(cost), sess.run(W), sess.run(b))