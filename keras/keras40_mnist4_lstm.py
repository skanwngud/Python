# lstm model (N, 28, 28) -> (N, 28, 28, 1) -> (N, 764, 1) = (N, 28*14, 2) = (N, 28*28, 1)
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=55)


x_train=x_train.reshape(x_train.shape[0], 14*7, 8)/255.
x_test=x_test.reshape(x_test.shape[0], 14*7, 8)/255.
x_val=x_val.reshape(x_val.shape[0], 14*7, 8).astype('float32')/255.

# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2], 1)
# x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2],1)
# x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2], 1)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

model=Sequential()
model.add(LSTM(150, activation='relu', input_shape=(14*7, 8)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='softmax'))

early=EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
pred=model.predict(x_test[:5])

print(loss)
print(pred)
print(np.argmax(pred, axis=-1))

# results - x.shape=28*28, 1
# 1 epoch 당 시간이 10분이 넘어가 모델을 돌리기 어려웠습니다....

# results - x.shape=7*7, 16 - 6min
# [0.9092453122138977, 0.6797000169754028]
# [[1.0782057e-04 8.8766692e-03 1.2210253e-02 9.8280817e-02 2.0893561e-03
#   1.1492911e-01 1.2999703e-04 6.8295544e-01 6.5321098e-03 7.3888414e-02]
#  [2.4906138e-03 2.0817837e-02 7.9818606e-01 1.1802545e-01 1.2904021e-04
#   4.2638175e-02 1.1522792e-02 5.4297587e-03 6.9076486e-04 6.9577334e-05]
#  [1.5828136e-04 8.5612994e-01 1.6878283e-02 3.0498035e-02 1.2823683e-03
#   2.4623420e-02 1.5461980e-02 2.7818138e-02 2.6373971e-02 7.7553216e-04]
#  [5.0797641e-01 7.1240156e-03 1.3115208e-01 4.8002467e-02 3.0049305e-02
#   3.2136831e-02 1.4227022e-01 1.4812499e-02 5.5769801e-02 3.0706419e-02]
#  [1.1574068e-03 4.7376037e-05 1.0721060e-03 9.5271862e-05 9.6430945e-01
#   1.8524904e-03 2.1704253e-02 6.4018060e-04 2.8019948e-03 6.3194144e-03]]
# [7 2 1 0 4]

# results - x.shape=14*7, 8 - 12min
# [0.5308992266654968, 0.7982000112533569]
# [[4.2896082e-07 2.8750826e-05 7.5017900e-04 7.9475958e-03 1.6826526e-08
#   1.3368070e-02 5.9739329e-09 9.7654188e-01 1.8465544e-04 1.1783842e-03]
#  [3.3085223e-03 4.6388619e-02 6.4963228e-01 9.6512541e-02 5.7982292e-04
#   1.0592466e-01 8.2520775e-02 7.3571713e-03 6.7335051e-03 1.0421714e-03]
#  [5.9014862e-08 9.9640805e-01 2.0722146e-05 5.6768255e-04 4.6634723e-06
#   1.5239466e-03 6.5947097e-05 7.0941582e-04 6.8670436e-04 1.2813149e-05]
#  [9.5426840e-01 1.0620539e-05 1.2627891e-02 1.5301185e-03 1.1171831e-03
#   2.0491264e-03 2.1540051e-02 4.5566336e-04 5.8273757e-03 5.7351269e-04]
#  [1.7589300e-04 6.6450339e-05 3.2253745e-03 3.4122475e-04 9.6671402e-01
#   9.7319047e-04 2.2035728e-03 1.8413119e-03 4.4407751e-03 2.0018099e-02]]
# [7 2 1 0 4]