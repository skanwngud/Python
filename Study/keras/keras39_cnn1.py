from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten # convolution 2D layer

model=Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(10,10,1)))
model.add(Flatten()) # Conv2D 를 Dense 와 엮기 위해서 Flatten 을 통과시켜서 2차원화(평탄화)시킴
model.add(Dense(1))

model.summary()