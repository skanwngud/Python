# VGG16 이용
# c:/data/image/data/male, female



import numpy as np
import tensorflow
import PIL.Image as image

from keras.applications import VGG16, VGG19, ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, BatchNormalization, Activation
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# 모델 정의
vgg16=VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(64, 64, 3)
)

# 데이터셋 준비
datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2
)

datagen2=ImageDataGenerator()

# 데이터 분리
# img_train=datagen.flow_from_directory(
#     'c:/data/image/data/',
#     target_size=(64, 64),
#     batch_size=1389,
#     subset='training',
#     class_mode='binary'
# )

# img_val=datagen.flow_from_directory(
#     'c:/data/image/data/',
#     target_size=(64, 64),
#     batch_size=347,
#     subset='validation',
#     class_mode='binary'
# )

vgg16.trainable=False # weights 만 가져옴

# print(type(img_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(img_train[0])

# numpy 저장
# np.save('c:/data/image/data/vgg16_x_train.npy', arr=img_train[0][0])
# np.save('c:/data/image/data/vgg16_y_train.npy', arr=img_train[0][1])
# np.save('c:/data/image/data/vgg16_x_val.npy', arr=img_val[0][0])
# np.save('c:/data/image/data/vgg16_y_val.npy', arr=img_val[0][1])

# numpy 로드
x_train=np.load('c:/data/image/data/vgg16_x_train.npy')
y_train=np.load('c:/data/image/data/vgg16_y_train.npy')
x_val=np.load('c:/data/image/data/vgg16_x_val.npy')
y_val=np.load('c:/data/image/data/vgg16_y_val.npy')

# print(type(x_train))
# print(type(x_train[0]))
# print(x_train.shape) # (1389, 64, 64, 3)
# print(y_train.shape) # (1389, )

x_train, x_test, y_train, y_test=train_test_split(
    x_train, y_train,
    train_size=0.9,
    random_state=23
)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

# print(y_train[0])

model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics='acc'
)

es=EarlyStopping(
    patience=50,
    verbose=1
)

rl=ReduceLROnPlateau(
    patience=30,
    verbose=1
)

mc=ModelCheckpoint(
    'c:/data/modelcheckpoint/keras81_male_female.hdf5',
    verbose=1,
    save_best_only=True,
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=500,
    steps_per_epoch=len(x_train),
    callbacks=[es, rl, mc]
)

loss=model.evaluate(
    x_test, y_test
)

model.load_weights('c:/data/modelcheckpoint/keras81_male_female.hdf5')

pred=model.predict(
    x_test
)

print('loss : ', loss[0])
print('acc : ', loss[1])

print(np.argmax(pred[:5], axis=-1))
print(np.argmax(y_test[:5], axis=-1))


model=load_model(
    'c:/data/modelcheckpoint/keras81_male_female.hdf5'
)

my=image.open(
    'c:/data/image/data/my.jpg'
)

iu=image.open(
    'c:/data/image/data/IU.png'
)

ma=image.open(
    'c:/data/image/data/ma.png'
)

hy=image.open(
    'c:/data/image/data/hyun.png'
)

yr=image.open(
    'c:/data/image/data/yr.png'
)

pix=[my, iu, ma, hy, yr]

a=list()
b=0
for i in pix:
    b+=1
    temp=list()
    temp=np.asarray(i)
    i=np.resize(
        i,
        (1, 768, 768, 3)
    )
    i=np.reshape(
        i,
        (1, 768, 768, 3)
    )/255.
    i=datagen.flow(
        i
    )
    print(str(b) + ' 번째 사진이 남자일 확률 : ', np.round(model.predict(i)[0][0]*100, 2), '% 입니다.')
    if model.predict(i)[0][0]>0.5:
        print(str(b) + ' 번째 사진은 남자입니다.')
    else:
        print(str(b) + ' 번째 사진은 여자입니다.')

# results
# Epoch 00031: early stopping
# loss :  0.0
# acc :  0.4932493269443512

# vgg16
# [2.070282459259033, 0.7194244861602783]
# [0 0 1 0 1]
# [0 0 1 0 1]

# vgg19
# [2.108022928237915, 0.7410072088241577]
# [0 0 1 0 1]
# [0 0 1 0 1]

# ResNet50
# loss :  3.8408255577087402
# acc :  0.5611510872840881
# [1 1 1 1 1]
# [0 0 1 0 1]