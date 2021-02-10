# 나를 찍어서 남자인지 여자인지 구별
# acc도 나오게끔

# fit_generator 사용

import tensorflow
import numpy as np

from PIL import Image # 내 사진 불러오기 위해 임포트

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# male = 841
# female = 895

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    vertical_flip=True,
    horizontal_flip=True,
    zoom_range=0.7,
    rescale=1./255,
    validation_split=0.2,
    shear_range=1.2
)

datagen2=ImageDataGenerator(rescale=1./255)

# data
train=datagen.flow_from_directory(
    'c:/data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=64,
    subset="training"
)

val=datagen.flow_from_directory(
    'c:/data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=64,
    subset="validation"
)

# 내 사진 관련
im=Image.open(
    'c:/data/image/data/my.jpg' # 내 사진 로드
)

iu=Image.open(
    'c:/data/image/data/IU.png'
)

my=np.asarray(im) # numpy 화
my=np.resize( # (4032, 3024, 3) 크기를 (128, 128, 3) 으로 줄임
    my,
    (128, 128, 3)
)
my=my.reshape(
    1, 128, 128, 3 # fit, predict 하기 위해 reshape
)

IU=np.asarray(iu)
IU=np.resize(
    im,
    (128, 128, 3)
)
IU=IU.reshape(
    1, 128, 128, 3
)

predict=datagen2.flow(my)
iu_pred=datagen2.flow(IU)

# model
model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(3, padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(2, padding='same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')
cp=ModelCheckpoint(
    filepath='c:/data/modelcheckpoint/keras67_my_{val_acc:.4f}_{val_loss:.4f}.hdf5',
    monitor='val_acc',
    save_best_only=True,
    verbose=1
)

# loss, acc 출력을 위한 history
history = model.fit_generator(
    train,
    steps_per_epoch=22,
    epochs=300,
    callbacks=[es, rl, cp],
    validation_data=val
)

# predict 결과값을 0, 1 로 이진화
pred=model.predict(predict)
pred_iu=model.predict(iu_pred)

pix=[pred, pred_iu]

print(pred)
print(pred_iu)

print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

# pred 값을 남성, 여성으로 분류하여 출력
for i in pix:
    if i>0.5:
        print('남성', pred[0][0])

    else:
        print('여성', pred[0][0])

# results
# [[1]]
# loss :  0.5185294151306152
# acc :  0.7343412637710571
# 남성
