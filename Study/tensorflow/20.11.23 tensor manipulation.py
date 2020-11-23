import tensorflow as tf
import numpy as np
sess=tf.Session()
# array 및 slicing
t=np.array([0, 1, 2, 3, 4, 5, 6])
print(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

p=np.array([[1,2,3],[4,5,6,],[7,8,9],[10, 11, 12]])
print(p)
print(p.ndim)
print(p.shape)

m1=tf.constant([[1,2],[3,4]])
m2=tf.constant([[1],[2]])
print("m1 shape", m1.shape)
print("m2 shape", m2.shape)
print(sess.run(tf.matmul(m1,m2)))

# Broadcasting
# - 보통 matrix 간의 계산은 차원이 맞아야하지만 Broadcasting 을 이용하면 차원을 맞추지 않아도 연산이 가능하다.
mat1=tf.constant([[1,2]])
mat2=tf.constant(3)
print(sess.run(mat1+mat2))

# Reshape
# - 변수의 shape 을 바꾸고 싶을 때 사용한다.
a=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
print(sess.run(tf.reshape(a, shape=([-1,3]))))
print(sess.run(tf.reshape(a, shape=([-1,1,3]))))
print(sess.run(tf.squeeze(a)))
print(sess.run(tf.expand_dims(a, 1)))