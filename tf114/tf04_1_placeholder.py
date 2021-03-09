import tensorflow as tf

node1=tf.constant(3.0, tf.float32)
node2=tf.constant(4.0)
node3=tf.add(node1, node2)

sess=tf.Session()

a = tf.placeholder(tf.float32) # input 의 개념
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(
    adder_node,
    feed_dict={a:3, b:4.5}
)) # 7.5 (placeholder 는 feed_dict 를 반드시 가지며, feed_dict 에서 값을 변경할 수 있다.)

print(sess.run(
    adder_node,
    feed_dict={a:[1,3], b:[3,4]}
)) # [4. 7.] (리스트의 형태로 값을 넣으면 리스트의 형태로 값이 출력 된다.)

add_and_triple = adder_node * 3

print(sess.run(
    add_and_triple,
    feed_dict={a:4, b:2}
)) # 18.0