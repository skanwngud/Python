import tensorflow as tf

x_data=[[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x_test=[[2,1,1],[3,1,2],[3,3,4]]
y_test=[[0,0,1],[0,0,1],[0,0,1]]

x=tf.placeholder(tf.float32, shape=[None,3])
y=tf.placeholder(tf.float32, shape=[None,3])
w=tf.Variable(tf.random_normal([3,3]), name='weight')
b=tf.Variable(tf.random_normal([3]), name='bias')

hypothesis=tf.nn.softmax(tf.matmul(x,w)+b)
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# learing_rate 가 너무 큰 수치이거나 작은 수치이면 학습이 제대로 되지 않음

prediction=tf.argmax(hypothesis, 1)
is_correct=tf.equal(prediction, tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range (201):
        cost_val, w_val, _ = sess.run([cost, w, optimizer], feed_dict={x:x_data, y:y_data})
        print(step, cost_val, w_val)

    print("prediction:", sess.run(prediction, feed_dict={x:x_test}))
    print("accuracy:", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))