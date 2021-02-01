# import libraries
import numpy as np
import pandas as pd

# read csv files
train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

# print(train.info()) # (2408, 783)
# print(pred.info()) # (20480, 784)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Dense

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

x=train.iloc[:, 2:] # Letter 제외
y=train.iloc[:, 0] # Digit 값
pred=pred.iloc[:, 1:] # Letter 제외

# numpy 전환
x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

x=x.reshape(-1, 28, 28, 1)/255.
pred=pred.reshape(-1, 28, 28, 1)

print(x.shape) # (2048, 784)
print(y.shape) # (2048, )

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=23)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

print(x_train.shape) # (1310, 784)
print(x_val.shape) # (328, 784)
print(x_test.shape) # (410, 784)
print(y_train.shape) # (1310, )
print(y_val.shape) # (328, )
print(y_test.shape) # (410, )

# pca=PCA(95)
# x_train=pca.fit_transform(x_train)
# x_test=pca.transform(x_test)
# x_val=pca.transform(x_val)

# cumsum=np.cumsum(pca.explained_variance_ratio_)
# print(np.argmax(cumsum>=0.95)+1) # 95

# print(x_train.shape) # (1310, 9

# model
input=Input(shape=(28, 28, 1))
conv2=Conv2D(128, 2, padding='same', activation='relu')(input)
conv2=Conv2D(256, 2, padding='same', activation='relu')(conv2)
max=MaxPooling2D(2, padding='same')(conv2)
conv2=Conv2D(128, 2, padding='same', activation='relu')(max)
conv2=Conv2D(256, 2, padding='same', activation='relu')(conv2)
max=MaxPooling2D(2, padding='same')(conv2)
conv2=Conv2D(128, 2, padding='same', activation='relu')(max)
conv2=Conv2D(64, 2, padding='same', activation='relu')(conv2)
flat=Flatten()(conv2)
dense=Dense(500, activation='relu')(flat)
output=Dense(10, activation='softmax')(dense)
model=Model(input, output)

# compile
es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
rl=ReduceLROnPlateau(monitor='acc', patience=5, mode='auto', verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=1000, batch_size=128, callbacks=[es, rl])

# eval, pred
loss=model.evaluate(x_test, y_test)
# y_pred=model.predict(pred)

# y_predict=pd.DataFrame(y_pred)
# y_predict.to_csv('../data/dacon/data/1.csv')

# print(loss)
# print(y_pred[0])
# print(pred[0])

# print(y_pred.shape)

# pred=pred.reshape(-1, 28, 28, 1)
sub['digit']=np.argmax(model.predict(pred), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples.csv', index=False)