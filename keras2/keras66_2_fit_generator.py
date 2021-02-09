import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 전처리 개념 (이미지 데이터 증폭)

train_datagen=ImageDataGenerator(
    rescale=1./255, # /255 전처리
    horizontal_flip=True, # 수평이동
    vertical_flip=True, # 수직이동
    width_shift_range=0.1, # 좌우 이동
    height_shift_range=0.1, # 상하 이동
    rotation_range=5, # 회전
    zoom_range=1.2, # 확대/축소
    shear_range=0.7, # 당겨서 변형
    fill_mode='nearest'
    # nearest : 이미지 이동 후 생긴 공백을 주변값과 비슷한 값으로 채움
    # 0 : 공백을 0으로 채움
)
test_datagen=ImageDataGenerator(rescale=1./255) # test set은 크기 전처리만 시킴

# 데이터를 변환 시켜주는 부분 flow 또는 flow_from_directory
# flow_from_directory : 폴더 자체에 접근해서 처리함

# train_generator
xy_train=train_datagen.flow_from_directory(
    'c:/data/image/brain/train', # 경로지정 - ad, normal 폴더 2개의 전체 이미지를 전부 데이터화 시킴
    target_size=(150, 150), # 수치는 임의대로 가능
    batch_size=5, # 전체 크기보다 큰 수치를 주어도 최대 수치로 나오게 된다 (이 경우엔 160)
    class_mode='binary' # 앞의 데이터가 0 이 되면 뒤의 데이터가 자동으로 1이 된다
)

# Found 160 images belonging to 2 classes.

# flow_from_directory 를 통해서 이미지를 데이터화 시킴
# ad x - (80, 150, 150, 1) - x값은 0~1 (rescale 을 시킴)
# ad y - (80, ) - y값은 전부 0
# normal x - (80, 150, 150, 1)
# normal y - (80, ) - y값은 전부 1

# test_generator
xy_test=test_datagen.flow_from_directory(
    'c:/data/image/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)

# Found 120 images belonging to 2 classes.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten,\
    BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model=Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(32, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(32, 3, padding='same'))
model.add(Conv2D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(3, padding='same'))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(3, padding='same'))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

es=EarlyStopping(patience=100, verbose=1)
rl=ReduceLROnPlateau(patience=20, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# fit_generator
history=model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=1000,
    validation_data=xy_test, validation_steps=4, callbacks=[es, rl]
)
# validaton_steps 찾아볼 것
# steps_per_epoch = 전체 데이터 갯수를 batch_size 로 나눴을 때의 값을 넣어준다.
# e.g. 160개의 데이터를 5의 batch_size 로 나눈 값이 32 이므로 32로 넣어줘야한다.
# 그 값의 미만인 경우 (31) 전체 학습을 31 번만 진행한다.
# 만약 32 로 다 떨어지지 않는 경우 (e.g. 161) 에는 161/5= 32.2 이므로 0.2 분의 훈련도
# 진행해야하기 때문에 1을 더 추가시켜 33을 최적의 epochs 로 넣어준다.

# fit 은 x,y 를 따로 넣어주지만 fit_generator 은 flow, flow_from_directory 을 통과한 데이터 자체를 넣어준다.

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

# visualization

import matplotlib.pyplot as plt

plt.plot(acc)
plt.plot(val_acc)
plt.plot(loss)
plt.plot(val_loss)

plt.ylabel('acc, loss')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss'])

plt.show()

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])

# results
