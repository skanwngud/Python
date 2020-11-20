import tensorflow as tf

sess=tf.Session()
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:[1], b:[2]}))
print(sess.run(adder_node, feed_dict={a:[1,2], b:[3,4]}))