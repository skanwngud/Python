import tensorflow as tf
print(tf.__version__) # 1.14.0

hello=tf.constant("Hello world") # constant : 상수, placeholder : 입력값만 받음, variable : 변수
sess=tf.Session()

print(sess.run(hello))
# 항상 sess.run 을 실행시켜야 그 값의 내용물이 출력 됨
# sess.run 을 하지 않고 print(hello) 를 하게 되면 단순 그 자료의 형태만 나온다
