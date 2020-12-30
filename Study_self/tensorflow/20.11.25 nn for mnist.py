import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate=0.001
training_epochs=15
batch_size=100

x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32, [None, 10])

w1=tf.Variable(tf.random_normal([784, 256]))
b1=tf.Variable(tf.random_normal([256]))
l1=tf.nn.relu(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.random_normal([256, 256]), name='weight2')
b2=tf.Variable(tf.random_normal([256]), name='bias2')

l2=tf.nn.relu(tf.matmul(l1,w2)+b2)

w3=tf.Variable(tf.random_normal([256, 10]), name='weight3')
b3=tf.Variable(tf.random_normal([10]), name='bias3')

hypothesis=tf.matmul(l2,w3)+b3

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct=tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys= mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})
            avg_cost+=cost_val/total_batch

        print("epoch: {:04d}, cost: {:.9f}".format(epoch+1, avg_cost))
    print("learning finish")

    print("accuracy:", accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

    r=random.randint(0, mnist.test.num_examples-1)
    print("labelL:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={x:mnist.test.images[r:r+1]}))
