# 실습 : 텐서의 사칙연산 만들 것

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1, node2)
node4 = tf.multiply(node1, node2) # 곱하기
node5 = tf.subtract(node1, node2) # 빼기
node6 = tf.divide(node1, node2) # 나누기
node7 = tf.mod(node1, node2) # 나눗셈의 나머지

sess = tf.Session()

print('덧셈 : ', sess.run(node3))
print('뺄셈 : ', sess.run(node5))
print('곱셈 : ', sess.run(node4))
print('나눗셈 : ', sess.run(node6))
print('나머지 : ', sess.run(node7))