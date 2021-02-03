## Conv2D, train_test_split


# import libraries
import numpy as np
import pandas as pd

# read csv files
train=pd.read_csv('../data/dacon/data/train.csv')
test1=pd.read_csv('../data/dacon/data/test.csv')
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

# print(train.info()) # (2408, 783)
# print(pred.info()) # (20480, 784)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape,\
        Dense, BatchNormalization, Activation, concatenate, Concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

x=train.drop(['id', 'letter', 'digit'], axis=1) # Letter 제외
y=train['digit'].values # Digit 값
pred=test1.drop(['id', 'letter'], axis=1) # Letter 제외

x_letter=train['letter'].values # Letter 값
y_letter=train['letter'].values # Letter 값
# pred_letter=pred.iloc[:, 0] # Letter 값

pred_letter=test1['letter'].values

# print(type(y))

# numpy 전환
x=x.to_numpy()
# y=y.to_numpy()
pred=pred.to_numpy()

print(x.shape) # (2048, 784)
print(y.shape) # (2048, )

print(y_letter.shape)

print(pred.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)
# x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=23)

x_l_train, x_l_test, y_l_train, y_l_test=train_test_split(x_letter, y_letter, train_size=0.8, random_state=23)

x_train=x_train.reshape(-1, 28, 28, 1)/255.
x_test=x_test.reshape(-1, 28, 28, 1)/255.
# x_val=x_val.reshape(-1, 28, 28, 1)/255.
# pred=pred.reshape(-1, 28, 28, 1)/255.
pred=pred/255.
pred_letter=pred_letter.reshape(-1, 1)

# print(pred[1, 0])
# print(pred.shape) # (20480, 784)
# print(pred_letter.shape) # (20480, 1)
# print(pred_letter[0,0]) # L


x_l_train=x_l_train.reshape(-1, 1, 1, 1)
x_l_test=x_l_test.reshape(-1, 1, 1, 1)
# pred_letter=pred_letter.reshape(-1, 28, 28, 1)/255.

# y_l_train=y_l_train.reshape(-1, 1)
# y_l_test=y_l_test.reshape(-1, 1)

# enc=OneHotEncoder()
# y_l_train=enc.fit_transform(y_l_train).toarray()
# y_l_test=enc.transform(y_l_test).toarray()
# pred_letter=enc.transform(pred_letter).toarray()

y_l_train=y_l_train.reshape(-1, 1)
y_l_test=y_l_test.reshape(-1, 1)
pred_letter=pred_letter.reshape(-1, 1)

# cumsum=np.cumsum(pca.explained_variance_ratio_)
# print(np.argmax(cumsum>=0.95)+1) # 95

# print(x_train.shape) # (1310, 9)

predict=np.append(pred, pred_letter, axis=1)

print(x_train.shape) # (1638, 28, 28, 1)
print(x_test.shape) # (410, 28, 28, 1)
print(x_l_train.shape) # (1638, 1, 1, 1)
print(x_l_test.shape) # (410, 1, 1, 1)

print(y_train.shape) # (1638, )
print(y_test.shape) # (410, )
print(y_l_train.shape) # (42588, 1)
print(y_l_test.shape) # (10660, 1)

print(pred.shape) # (20480, 784)
print(pred_letter.shape) # (20480, 1)

print(predict.shape) # (20480, 785)


# model
input=Input(shape=(28, 28, 1))
conv2=Conv2D(64, 2, padding='same')(input)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2)(act)
conv2=Conv2D(128, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(256, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(512, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(1024, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(1024, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
conv2=Conv2D(512, 2, padding='same')(max)
bn=BatchNormalization()(conv2)
act=Activation('relu')(bn)
max=MaxPooling2D(2, padding='same')(act)
flat=Flatten()(max)
dense=Dense(1000, activation='softmax')(flat)
bn=BatchNormalization()(dense)
output=Dense(1)(bn)
model=Model(input, output)

input2=Input(shape=(1, 1, 1))
conv22=Conv2D(64, 2, padding='same')(input2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(128, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(256, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(512, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(1024, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(1024, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
conv22=Conv2D(512, 2, padding='same')(max2)
bn2=BatchNormalization()(conv22)
act2=Activation('relu')(bn2)
max2=MaxPooling2D(2, padding='same')(act2)
flat2=Flatten()(max2)
dense2=Dense(1000, activation='relu')(flat2)
bn2=BatchNormalization()(dense2)
output2=Dense(1, activation='relu')(bn2)
model2=Model(input2, output2)

# merge=concatenate([output, output2])
# middle=Dense(256, activation='relu')(merge)
# middle=Dense(10, activation='softmax')(middle)
# model=Model([input, input2], middle)


# compile
es=EarlyStopping(monitor='val_loss', patience=100, mode='auto')
rl=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_2_{val_acc:.4f}-{val_loss:.4f}.hdf5',
                        monitor='val_acc', mode='auto', save_best_only=True)
cp2=ModelCheckpoint(filepath='../data/modelcheckpoint/dacon_22_{val_acc:.4f}-{val_loss:.4f}.hdf5',
                        monitor='val_acc', mode='auto', save_best_only=True)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_test, y_test),
        epochs=100, batch_size=128, callbacks=[es, rl, cp])
model2.compile(loss='mse', optimizer='adam', metrics='acc')
model2.fit(x_l_train, y_l_train, validation_data=(x_l_test, y_l_test),
        epochs=100, batch_size=128, callbacks=[es, rl, cp2])


# eval, pred
# loss=model.evaluate(x_test, y_test)
# y_pred=model.predict(pred)

# y_predict=pd.DataFrame(y_pred)
# y_predict.to_csv('../data/dacon/data/1.csv')

# print(loss)
# print(y_pred[0])
# print(pred[0])

# print(y_pred.shape)

# pred=pred.reshape(-1, 28, 28, 1)
sub['digit']=np.argmax(model.predict(predict), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples.csv', index=False)
