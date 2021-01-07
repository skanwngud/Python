import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# 2. modeling

model=Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# save model
model.save('./Study/model/save_keras35.h5')
model.save('.//Study//model//save_keras35_1.h5')
model.save('.\Study\model\save_keras35_2.h5')
model.save('.\\Study\\model\\save_keras35_3.h5')

# 경로명에 . 하나만 찍으면 현재 폴더를 지정함 (현재 폴더 = Study, 현재 작업중인 keras 폴더가 아님)
# 확장자명은 h5
