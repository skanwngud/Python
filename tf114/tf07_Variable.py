import tensorflow as tf

tf.compat.v1.set_random_seed(23)

W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

print(W) # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
a = sess.run(W)
print('a : ', a) # [0.13260844]
sess.close()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
b = W.eval()
print('b : ', b) # [0.13260844]
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
c = W.eval(session=sess)
print('c : ', c) # c :  [0.13260844]
sess. close()