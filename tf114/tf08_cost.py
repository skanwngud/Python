import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [3., 5., 7.]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y))

w_history = list()
cost_histhory = list()

with tf.compat.v1.Session() as sess : 
    for i in range(-30, 50):
        curr_w = i * 0.1
        curr_cost = sess.run(
            cost, feed_dict={w : curr_w}
        )

        w_history.append(curr_w)
        cost_histhory.append(curr_cost)

print('='*50)
print(w_history)
print('='*50)
print(cost_histhory)
print('='*50)

plt.plot(w_history, cost_histhory)
plt.show()

# 위의 경우 initializer 를 쓰지 않았는데도 코드가 돌아가는데 그 이유는,
# (추측하기로는) 위의 cost, hypothesis 등이 tensorflow 의 변수가 아닌 python 의 변수로 판단해서 그렇다