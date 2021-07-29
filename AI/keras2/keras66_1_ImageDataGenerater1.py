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
    batch_size=1200, # 전체 크기보다 큰 수치를 주어도 최대 수치로 나오게 된다 (이 경우엔 160)
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

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000018528EA8550>
# numpy 는 한가지의 데이터 셋만 가지고 있으며 2개 이상일 경우 list, dictionaryd 의 형태를 띈다

print(xy_train[0])
# x, y 가 동시에 출력
# 이대로 출력하면 x, y 값이 5개가 출력이 되는데 batch_size 의 영향을 받음

print(xy_train[0][0]) # x 값
print(xy_train[0][0].shape) # (5, 150, 150, 3)

print(xy_train[0][1]) # y 값 [0. 0. 1. 0. 0.]
print(xy_train[0][1].shape) # (5,)
# batch_size=10 으로 주게 된다면 총 160개의 데이터이므로 0~15 까지
# batch_size=5 이면 0~31 까지 ...
# xy_train[15][1] 에서 [15] 는 x 값 [1] 은 y 값을 나타낸다

# fit, fit_generator
