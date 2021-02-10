
# imagedatagenerator 를 통과할 때 폴더구조가 female, male 순이었으므로,
# female 이 0, male 이 1 로 치환이 된다.
# 이때 sigmoid 를 지나게 되는데 0에서 1 '사이'의 값이 나오기 때문에,
# 상대적으로 0 에 가까우면 female, 1에 가까우면 male 이다.

# 따라서, 0.5 를 기준으로 그보다 낮으면 female, 높으면 male 이다.

import tensorflow
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from sklearn.metrics import accuracy_score

x_train=np.load('c:/data/image/data/train_set.npy')
y_train=np.load('c:/data/image/data/test_set.npy')
x_val=np.load('c:/data/image/data/val_set.npy')
y_val=np.load('c:/data/image/data/val_test_set.npy')

datagen=ImageDataGenerator()

print(x_train.shape) # (1389, 128, 128, 3)
print(y_train.shape) # (1389, )
print(x_val.shape) # (347, 128, 128, 3)
print(y_val.shape) # (347, )

# x_val=x_val.reshape(-1, 1)
# y_val=y_val.reshape(-1, 1)

model=load_model(
    'c:/data/modelcheckpoint/keras67_my_0.7810_0.4785.hdf5'
)

my=Image.open(
    'c:/data/image/data/my.jpg'
)

iu=Image.open(
    'c:/data/image/data/IU.png'
)

ma=Image.open(
    'c:/data/image/data/ma.png'
)


hy=Image.open(
    'c:/data/image/data/hyun.png'
)

yr=Image.open(
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
        (1, 128, 128, 3)
    )
    i=np.reshape(
        i,
        (1, 128, 128, 3)
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
# 1 번째 사진이 남자일 확률 :  54.11 % 입니다.
# 1 번째 사진은 남자입니다.
# 2 번째 사진이 남자일 확률 :  58.45 % 입니다.
# 2 번째 사진은 남자입니다.
# 3 번째 사진이 남자일 확률 :  40.09 % 입니다.
# 3 번째 사진은 여자입니다.
# 4 번째 사진이 남자일 확률 :  46.09 % 입니다.
# 4 번째 사진은 여자입니다.

# results
