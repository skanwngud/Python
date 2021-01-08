from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D # convolution 2D layer

model=Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1,
                padding='same', input_shape=(10,10,1)))
# padding = 'same' -> Conv2D 를 지나면서 줄어드는 shape 를 줄이지 않는다 default = 'valid'
# strides = 2 -> 특성 추출시 2칸씩 움직이며 추출함, kernel_szie 와 동일시 중첩되는 부분이 거의 없음. default = 1
model.add(MaxPooling2D(pool_size=(2,3)))
# Maxpooling 을 거치게 되면 사이즈가 줄어들음 (특성추출을 위함), default=(2,2)
# 크기 조절 가능하며 가로세로가 달라도 작용 (2, 3) 형태도 가능
# Conv 이후에 사용 가능
model.add(Conv2D(9, (2,2), padding='valid'))
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2)) # kernel_size=(2,2) 를 2 라고 인식 가능함
model.add(Flatten()) # Conv2D 를 Dense 와 엮기 위해서 Flatten 을 통과시켜서 2차원화(평탄화)시킴
model.add(Dense(1))
# LSTM 과는 달리 CNN 은 특성을 추출하는 것이기 때문에 레이어를 여러개 쌓을수록 좋다


model.summary()
# kernel_size (2*2) input channel (1) filter (10) bias (1)
# (kernel_size * channel + bias)* filter
# ((2*2)*1+1)10 = 50

# case of color (RGB channel =3)
# ((2*2)*3+1)*10=130