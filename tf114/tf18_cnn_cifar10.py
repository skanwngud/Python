import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
tf.set_random_seed(12)

# 1. data

(x_train, y_train), (x_test, y_test) = \
    cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28, 28, 1)/255.
x_test = x_test.reshape(-1, 28, 28, 1)/255.

print(x_train.shape)
print(y_train.shape)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000/100

x = tf.placeholder(
    tf.float32,
    [None, 28, 28, 1]
)

y = tf.placeholder(
    tf.float32,
    [None, 10]
)

# 2. model
# w1
w1 = tf.get_variable(
    'w1', shape = [3, 3, 1, 32] # 32 : filter (output node)
)

b1 = tf.Variable(
    tf.random_normal([32]),
    name = 'bias1'
)

L1 = tf.nn.conv2d(
    x, w1, # x, w1 둘 다 4차원으로 서로 크기를 맞춰야한다
    strides = [1,1,1,1], # 2칸씩 띄우려면 [1, 2, 2, 1] : 앞 뒤 1 은 행렬의 크기를 맞추기 위함
    padding='SAME'
) # Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

print(L1)
# Conv2D(filter, kernel_size, input_shape)
# Conv2D(10, (2,2), input_shape=(7, 7, 1))

L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # (?, 14, 14, 32)
print(L1) # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
L1 = tf.nn.dropout(L1, keep_prob=0.7)


# w2
w2 = tf.get_variable(
    'w2', shape = [3, 3, 32, 64] # 첫 번째 레이어의 두 아웃풋이 두 번째 레이어의 인풋이 되므로 32, 32, ?, ?2 에서
)                                # ? 값은 L1 의 아웃풋 값은 32가 된다

b2 = tf.Variable(
    tf.random_normal([64]),
    name = 'bias2'
)

L2 = tf.nn.conv2d(
    L1, w2,
    strides=[1,1,1,1],
    padding='SAME'
)

L2 = tf.nn.selu(L2)

L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2) # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
L2 = tf.nn.dropout(L2, keep_prob=0.7)


# w3
w3 = tf.get_variable(
    'w3', shape = [3, 3, 64, 128]
)

b3 = tf.Variable(
    tf.random_normal([128]),
    name = 'bias3'
)

L3 = tf.nn.conv2d(
    L2, w3,
    strides=[1,1,1,1],
    padding='SAME'
)

L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(
    L3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME'
)
print(L3) # Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
L3 = tf.nn.dropout(L3, keep_prob=0.7)


# w4
w4 = tf.get_variable(
    'w4', shape = [3,3,128, 64]
)

b4 = tf.Variable(
    tf.random_normal([64]),
    name = 'bias4'
)

L4 = tf.nn.conv2d(
    L3, w4,
    strides=[1,1,1,1],
    padding='SAME'
)

L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(
    L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'
)
print(L4) # Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

L_flat = tf.reshape(L4, [-1, 2*2*64]) # 행 빼고 전부 곱해준다
print("Flatten : ", L_flat)

# w5
w5 = tf.get_variable(
    'w5', shape = [256, 64],
    initializer=tf.contrib.layers.xavier_initializer()
)

b5 = tf.Variable(
    tf.random_normal([64]),
    name = 'bias5'
)

L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=0.7)

# w6
w6 = tf.get_variable(
    'w6', shape = [64, 32],
    initializer=tf.contrib.layers.xavier_initializer()
)

b6 = tf.Variable(
    tf.random_normal([32]),
    name = 'bias6'
)

L6 = tf.nn.selu(tf.matmul(L5, w6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=0.7)

# w7
w7 = tf.get_variable(
    'w7', shape=[32, 10],
    initializer=tf.contrib.layers.xavier_initializer()
)

b7 = tf.Variable(
    tf.random_normal([10]),
    name = 'bias7'
)

hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)

# compile
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis = 1)) # categorical_crossentropy
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# train
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
            [loss, train],
            feed_dict=feed_dict
        )
        avg_cost += c/total_batch
    print('Epoch : ', '%04d' %(epoch + 1),
          'cost = {:.9f}'.format(avg_cost))

print('training finished!!')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
print('Acc : ', sess.run(
    accuracy,
    feed_dict={x:x_test, y:y_test}
))