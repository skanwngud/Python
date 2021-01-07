import numpy as np

a=np.array(range(1,11))
size=5
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

def split_x (seq, size):
    n=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:i+size]
        n.append(subset)
    return np.array(n)

datasets=split_x(a, size)

print(datasets)

x=datasets[:, :4]
y=datasets[:, -1]

print(x.shape)
print(y.shape)

x=x.reshape(x.shape[0], x.shape[1], 1)

model=load_model('./Study/model/save_keras35.h5')
model.add(Dense(5, name='kingkeras1')) # layer name : dense
model.add(Dense(1, name='kingkeras2')) # layer name : dense_1
# 저장 된 모델 안에 있는 레이어의 이름과 현재 시작 되는 레이어의 이름이 같아 충돌 됨
model.summary()


model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

loss=model.evaluate(x,y)
pred=model.predict(x)

print(loss)
print(pred)

# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
