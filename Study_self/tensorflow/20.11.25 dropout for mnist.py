import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST.data/", one_hot=True)

learning_rate=0.001
training_epochs=15
batch_size=100
total_batch=int(mnist.train.num_examples/batch_size)

x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32, [None, 10])

keep_prob=tf.placeholder(tf.float32)

w1=tf.get_variable('w1', shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([512]))
l1=tf.nn.relu(tf.matmul(x,w1)+b1)
l1=tf.nn.dropout(l1, keep_prob=keep_prob)

w2=tf.get_variable('w2', shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.random_normal([512]))
l2=tf.nn.relu(tf.matmul(l1,w2)+b2)
l2=tf.nn.dropout(l2, keep_prob=keep_prob)

w3=tf.get_variable('w3', shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b3=tf.Variable(tf.random_normal([512]))
l3=tf.nn.relu(tf.matmul(l2,w3)+b3)
l3=tf.nn.dropout(l3, keep_prob=keep_prob)

w4=tf.get_variable('w4', shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([512]))
l4=tf.nn.relu(tf.matmul(l3, w4)+b4)
l4=tf.nn.dropout(l4, keep_prob=keep_prob)

w5=tf.get_variable('w5', shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([10]))

hypothesis=tf.matmul(l4, w5)+b5

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        for i in range(total_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7}
            c, _=sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost+=c/total_batch
        print('epoch:', '%04d'% (epoch +1), 'cost:', '{:.9f}'.format(avg_cost))

    print('learning finish!!!')

    correct_prediction=tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy:', sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1}))

    r=random.randint(0, mnist.test.num_examples-1)
    print('label:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('prediction:', sess.run(tf.argmax(hypothesis, 1), feed_dict={x:mnist.test.images[r:r+1], keep_prob:1}))