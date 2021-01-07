import numpy as np

a=np.array(range(1,11))
size=5
from tensorflow.keras.models import load_model

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

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

loss=model.evaluate(x,y)
pred=model.predict(x)

print(loss)
print(pred)

# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
# 로드 된 모델이 컴파일이 되지 않았다는 에러메세지