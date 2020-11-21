import tensorflow as tf

x_data=[[1.2],[2.2],[3.2],[4.2]]
y_data=[[2.2],[3.2],[4.2],[5.2]]

x=tf.placeholder(tf.float32, shape=[None, 1])
y=tf.placeholder(tf.float32, shape=[None, 1])
w=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = w*x+b

cost=tf.reduce_mean(tf.square(hypothesis-y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range (2001):
        cost_val, w_val, b_val, _ = sess.run([cost, w, b, optimizer], feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, cost_val, w_val, b_val)