import tensorflow as tf
import matplotlib.pyplot as plt
# matplotlib 은 시각화 함수, 처음에 matplotlib.pyplot 을 import 한다.
import random
# 난수 생성.

from tensorflow.examples.tutorials.mnist import input_data
# "input_data"에 있는 mnist 의 데이터를 다운로드한다.

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
# MNIST_data 의 경로에 있는 mnist 데이터를 불러온다. one_hot 은 사용한다.

nb_classes=10
# classes 가 0~9 개 이므로 10.

x=tf.placeholder(tf.float32, shape=[None, 784])
y=tf.placeholder(tf.float32, shape=[None, nb_classes])
w=tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis=tf.nn.softmax(tf.matmul(x,w)+b)

cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct=tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs=15
batch_size=100
num_iterations=int(mnist.train.num_examples/batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        avg_cost=0
        for i in range(num_iterations):
            batch_xs, batch_ys= mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={x:batch_xs, y:batch_ys})
            avg_cost+=cost_val/num_iterations

        print("epoch: {:04d}, cost: {:.9f}".format(epoch+1, avg_cost))
    print("learning finish")

    print("accuracy:", accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

    r=random.randint(0, mnist.test.num_examples-1)
    print("labelL:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={x:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), camp="Greys", interpolation="nearest")
    # 이미지를 보여준다.
    plt.show()
    # matplotlib 의 함수 그래프를 보여준다.