import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

# 1. data
a=np.array(range(1,101))
size=5

def split_x (seq, size):
    n=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:i+size]
        n.append(subset)
    return np.array(n)

datasets=split_x(a, size)

print(datasets.shape)

x=datasets[:, :4]
y=datasets[:, -1]
print(x.shape, y.shape) # (96, 4) (96, )

x=x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (96, 4, 1)

# 2. modeling
model=load_model('./Study/model/save_keras35.h5')
model.add(Dense(5, name='kingkeras1'))
model.add(Dense(1, name='kingkeras2'))

from tensorflow.keras.callbacks import EarlyStopping

es=EarlyStopping(monitor='loss', patience=10, mode='auto')

# compile, fitting
model.compile(loss='mse', optimizer='adam', metrics='acc')
hist=model.fit(x,y,epochs=1000, batch_size=32, verbose=1, 
            validation_split=0.2, callbacks=es)

print(hist)
print(hist.history.keys()) # loss, acc, val_loss, val_acc
                           # val_loss 는 loss 보다 높거나, 두 간격이 너무 차이가 나면 신뢰도가 떨어짐
print(hist.history['loss']) # loss 값이 순서대로 출력 됨

# graph

import matplotlib.pyplot as plt
## 그래프의 지표 - 뭘 표현할 것인지
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

## 그래프의 x, y축
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc']) # 주석, default 는 그래프의 자동으로 빈 곳에 들어감
plt.show()