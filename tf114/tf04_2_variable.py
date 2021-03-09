import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.global_variables_initializer() # Variable 은 sess.run 을 들어가기 전 variables_initializer 를 통과 시켜줘야한다

sess.run(init)

print(sess.run(x)) # [2. ]