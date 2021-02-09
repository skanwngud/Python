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

# train_generator
xy_train=train_datagen.flow_from_directory(
    'c:/data/image/brain/train', # 경로지정 - ad, normal 폴더 2개의 전체 이미지를 전부 데이터화 시킴
    target_size=(150, 150), # 수치는 임의대로 가능
    batch_size=200, # 한 번에 해당 수치만큼의 이미지가 생성
    class_mode='binary' # 앞의 데이터가 0 이 되면 뒤의 데이터가 자동으로 1이 된다
    , save_to_dir='c:/data/image/brain_generator/train/'
)

# test_generator
xy_test=test_datagen.flow_from_directory(
    'c:/data/image/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
    , save_to_dir='.././data/image/brain_generator/test'
)

# save_to_dir 한 뒤 한 번 print 를 해줘야 생성이 됨
# batch_size * 해당 파일을 건드리는 횟수 (print, for, if ...)
print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][1].shape)
# print(xy_train[0][1])

# 과제 flow 를 사용