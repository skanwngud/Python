import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 784).astype('float32')/255
x_test = x_test.reshape(-1, 784)/255.

# print(x_train.shape) # (60000, 784)
# print(x_test.shape) # (10000, 784)
# print(y_train.shape) # (60000, 10)
# print(y_test.shape) # (10000, 10)

# 2. modeling
x = tf.placeholder(
    tf.float32,
    shape = [None, 784]
)

y = tf.placeholder(
    'float32',
    [None, 10]
)

# w1 = tf.Variable(
#     tf.random_normal([784, 100], stddev=0.1),
#     name = 'weight1'
# )

w1 = tf.get_variable(
    'weight1',
    shape = [784, 256],
    initializer = tf.initializers.he_normal() # kernel_initializer() : 가중치 초기화
)

print('w1 : ', w1) # w1 :  <tf.Variable 'weight1:0' shape=(784, 100) dtype=float32_ref>

b1 = tf.Variable(
    tf.random_normal([256], stddev=0.1),
    name = 'bias1'
)

print('b1 : ', b1) # b1 :  <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>

# w2 = tf.Variable(
#     tf.random_normal([100, 50], stddev=0.1),
#     name = 'weight2'
# )

w2 = tf.get_variable(
    'weight2',
    shape = [256, 128],
    initializer=tf.initializers.he_normal()
)

b2 = tf.Variable(
    tf.random_normal([128], stddev=0.1),
    name = 'bias2'
)

# w3 = tf.Variable(
#     tf.random_normal([50, 10], stddev=0.1),
#     name = 'weight3'
# )

w3 = tf.get_variable(
    'weight3',
    shape = [128, 64],
    initializer=tf.initializers.he_normal()
)

b3 = tf.Variable(
    tf.random_normal([64], stddev=0.1),
    name = 'bias3'
)

w4 = tf.get_variable(
    'weight4',
    shape = [64, 10],
    initializer=tf.initializers.he_normal()
)

b4 = tf.Variable(
    tf.random_normal([10]),
    name = 'bias4'
)

# layer1 = tf.nn.softmax(tf.matmul(x, w1) + b1) # softmax 는 중간에 끼면 성능이 제대로 안 나오는 경우가 많다
# layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.selu(tf.matmul(x, w1) + b1)
layer1 = tf.nn.elu(tf.matmul(x, w1) + b1) # relu, selu, elu 가능

print('layer 1 : ', layer1) # layer 1 :  Tensor("Elu:0", shape=(?, 100), dtype=float32)

# layer1 = tf.nn.dropout(layer1, keep_prob=0.3) # model.add(Dropout(0.3))

# print('layer1 : ', layer1) # layer1 :  Tensor("dropout/mul_1:0", shape=(?, 100), dtype=float32)


layer2 = tf.nn.selu(tf.matmul(layer1, w2) + b2)
# layer2 = tf.nn.dropout(layer2, keep_prob=0.3)

layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.dropout(layer3, keep_prob=0.3)

hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)


# 3. compile, fitting (multiple classification)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

training_epochs = 200
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000/100 = 600

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch): # 1 epoch 에 600 번
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run(
            [loss, train], # 포문을 돌면서 나오는 c 는 데이터 전체의 c 값이 아니라 데이터를 600분할 한 일부분에 대한 loss 이다
            feed_dict=feed_dict
        )
        avg_cost += c/total_batch # loss / 600 # 따라서 한 번 돌 때마다 나오는 loss 값을 전부 더한 뒤 600 을 나누면 loss 값의 평균이 나온다
        # 즉 avg_cost 는 1 epoch 당 loss 라고 판단할 수 있다
        # 위와 같은 경우 딱 나누어 떨어지지만 600001 / 100 과 같이 정확히 나누어지지 않는 경우엔 손실을 감수한다
    print('Epoch : ', '%04d' %(epoch + 1),
          'cost = {:.9f}'.format(avg_cost))

print('training finished!!')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
print('Acc : ', sess.run(
    accuracy,
    feed_dict={x:x_test, y:y_test}
))

# results
# Epoch :  0015 cost = 2.284967653
# training finished!!
# Acc :  0.3492

# Epoch :  0015 cost = 2.023979617
# training finished!!
# Acc :  0.4362

# Epoch :  0150 cost = 0.730361828
# training finished!!
# Acc :  0.7793

# Epoch :  0150 cost = 0.661845831
# training finished!!
# Acc :  0.8066

# Epoch :  0200 cost = 0.600975616
# training finished!!
# Acc :  0.8269

# Epoch :  0200 cost = 0.033295874
# training finished!!
# Acc :  0.9758

# Epoch :  0200 cost = 0.026092614
# training finished!!
# Acc :  0.9759