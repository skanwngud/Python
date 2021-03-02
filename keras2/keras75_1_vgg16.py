import tensorflow

from keras.applications import VGG16 # 레이어가 16개인 모델
from keras.layers import Dense, Flatten
from keras.models import Sequential

model=VGG16(
    weights='imagenet', # imagenet 에 있는 가중치를 가져옴
    include_top=False, # True 인 경우 imagenet 에 있었던 데이터 사이즈를 사용해야함 / False 로 해야 원하는 사이즈로 사용 가능
    input_shape=(32, 32, 3) # (224, 224, 3) 가 imagenet 에 있던 default 이미지 크기
)

# print(model.weights)

model.trainable=False # 훈련 시키지 않고 가중치만 그대로 사용한다 / default 는 True

model.summary()

print(len(model.weights)) # 26 - layer 가 16개지만 실제로 연산 되는 layer 는 13개
print(len(model.trainable_weights)) # 0 - 훈련하는 weights 값이 없으므로 0
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688


model.trainable=True # 모델도 훈련을 시킨다

model.summary()

print(len(model.weights)) # 26
print(len(model.trainable_weights)) # 26
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0