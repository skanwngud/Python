import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes=10

x=tf.placeholder(tf.float32, shape=[None, 784])
y=tf.placeholder(tf.float32, shape=[None, nb_classes])
w=tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis=tf.nn.softmax(tf.matmul(x,w)+b)

cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(cost)

is_correct=tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epoch=15
batch_size=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range (num_epoch):
        avg_cost=0
        total_batch=int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += c / total_batch

        print('epoch:', '%04d' %(epoch+1), 'cost =','{:.9f}'.format(avg_cost))
        print('accuracy:', accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

    r=random.randint(0, mnist.test.num_examples-1)
    print('label:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('prediction:', sess.run(tf.argmax(hypothesis, 1), feed_dict={x:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(-1, 28, 28), camp='Greys', interpolation='nearest')
    plt.show()