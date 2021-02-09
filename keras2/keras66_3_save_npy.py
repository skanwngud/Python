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
    batch_size=160, # fit 을 사용할 때엔 전체 데이터셋의 갯수만큼 준다
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
    batch_size=120,
    class_mode='binary'
)

# Found 120 images belonging to 2 classes.

print(xy_train)

print(xy_train[0])
print(xy_train[0][0]) # x 값
print(xy_train[0][0].shape) # (160, 150, 150, 3)

print(xy_train[0][1]) # y 값
print(xy_train[0][1].shape) # (160, )

# numpy 저장
np.save('c:/data/image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('c:/data/image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('c:/data/image/brain/npy/keras66_test_x.npy', arr=xy_test[0][0])
np.save('c:/data/image/brain/npy/keras66_test_y.npy', arr=xy_test[0][1])

# numpy 불러오기
x_train=np.load('c:/data/image/brain/npy/keras66_train_x.npy')
y_train=np.load('c:/data/image/brain/npy/keras66_train_y.npy')
x_test=np.load('c:/data/image/brain/npy/keras66_test_x.npy')
y_test=np.load('c:/data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160, )
print(x_test.shape, y_test.shape) # (120, 150, 150, 3) (120, )