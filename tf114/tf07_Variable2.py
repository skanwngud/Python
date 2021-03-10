import tensorflow as tf

x = [1, 2, 3]
W = tf.compat.v1.Variable([0.3], tf.float32)
b = tf.compat.v1.Variable([1.0], tf.float32)

hypothesis = W * x + b

# 실습
# 1. sess.run()
# 2. InteractiveSession()
# 3. sess.run() 없이

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
a = sess.run(hypothesis)
print('hypothesis : ', a)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
b = hypothesis.eval(session=sess)
print('hypothesis : ', b)
sess.close()

sess = tf.compat.v1.InteractiveSession()
sess.run = tf.compat.v1.global_variables_initializer()
c = hypothesis.eval()
print('hypothesis :', c)
sess.close()
