from tensorflow.python.framework.ops import disable_eager_execution # 즉시 실행

import tensorflow as tf


print(tf.executing_eagerly()) # False / 2점대에선 True

tf.compat.v1.disable_eager_execution() # 2점대에서 쓸 수 있는 코드
print(tf.executing_eagerly()) # False / 2점대에서도 False

hello=tf.constant('hello world')
# sess=tf.Session() 
sess=tf.compat.v1.Session() # 2.14 버전에서 사용

print(sess.run(hello))
