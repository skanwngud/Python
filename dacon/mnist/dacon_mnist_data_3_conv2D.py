## Conv2D, KFold

#  import libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt

# read csv files
train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

# print(train.info()) # (2408, 783)
# print(pred.info()) # (20480, 784)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Dense, BatchNormalization, Activation
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen2=ImageDataGenerator()

x=train.iloc[:, 2:] # Letter 제외
y=train.iloc[:, 0] # Digit 값
pred=pred.iloc[:, 1:] # Letter 제외

# numpy 전환
x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

print(x.shape) # (2048, 784)
print(y.shape) # (2048, )

x=x.reshape(-1, 28, 28, 1)/255.
pred=pred.reshape(-1, 28, 28, 1)/255.

y=to_categorical(y)

kf=KFold(n_splits=20, shuffle=True, random_state=22)

print(pred.shape)


for train_index, validation_index in kf.split(x, y):
    x_train=x[train_index]
    x_val=x[validation_index]
    y_train=y[train_index]
    y_val=y[validation_index]

    # train=datagen.flow(x_train, y_train, batch_size=128)
    # val=datagen.flow(x_val, y_val, batch_size=128)
    # pred=datagen.flow(pred, batch_size=128, shuffle=False)

    # x_train=x_train.reshape(-1, 28*28*1)
    # x_val=x_val.reshape(-1, 28*28*1)
    # pred=pred.reshape(-1, 28*28*1)

    # pca=PCA(277)
    # x_train=pca.fit_transform(x_train)
    # x_val=pca.transform(x_val)
    # pred=pca.transform(pred)

    # x_train=x_train.reshape(-1, 277, 1, 1)
    # x_val=x_val.reshape(-1, 277, 1, 1)
    # pred=pred.reshape(-1, 277, 1, 1)

    es=EarlyStopping(monitor='val_loss', patience=50, mode='auto')
    rl=ReduceLROnPlateau(monitor='val_loss', patience=25, mode='auto', verbose=1)
    cp=ModelCheckpoint(save_best_only=True, monitor='val_acc', mode='auto',
                        filepath='../data/modelcheckpoint/dacon_mnist_data_{val_acc:.4f}_{val_loss:.4f}.hdf5')

    # input=Input(shape=(28, 28, 1))
    # a=Conv2D(128, (2,2),  padding='same')(input)
    # a=BatchNormalization()(a)
    # a=Activation('relu')(a)
    # a=MaxPooling2D(2, padding='same')(a)
    # a_=Conv2D(128, (2,2), padding='same')(input)
    # a_=BatchNormalization()(a_)
    # a_=Activation('relu')(a_)
    # a_=MaxPooling2D(2, padding='same')(a_)
    # a2=a+a_
    # b=Conv2D(128, (2,2), padding='same')(input)
    # b=BatchNormalization()(b)
    # b=Activation('relu')(b)
    # b=MaxPooling2D(2, padding='same')(b)
    # b_=Conv2D(128, (2,2), padding='same')(input)
    # b_=BatchNormalization()(b_)
    # b_=Activation('relu')(b_)
    # b_=MaxPooling2D(2, padding='same')(b_)
    # b2=b+b_
    # c=a2+b2
    # d=Conv2D(256, (3,3), padding='same')(c)
    # d=BatchNormalization()(d)
    # d=Activation('relu')(d)
    # d=MaxPooling2D(3, padding='same')(d)
    # d_=Conv2D(256, (3,3), padding='same')(c)
    # d_=BatchNormalization()(d_)
    # d_=Activation('relu')(d_)
    # d_=MaxPooling2D(3, padding='same')(d_)
    # d2=d+d_
    # e=Conv2D(256, (3,3), padding='same')(c)
    # e=BatchNormalization()(e)
    # e=Activation('relu')(e)
    # e=MaxPooling2D(3, padding='same')(e)
    # e_=Conv2D(256, (3,3), padding='same')(c)
    # e_=BatchNormalization()(e_)
    # e_=Activation('relu')(e_)
    # e_=MaxPooling2D(3, padding='same')(e_)
    # e2=e+e_
    # f=d2+e2
    # flat=Flatten()(f)
    # dense=Dense(1024, activation='relu')(flat)
    # output=Dense(10, activation='softmax')(dense)
    # model=Model(input, output)

    # model.summary()

    model=Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Conv2D(128, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Conv2D(256, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(1024, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
    hist=model.fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=500, batch_size=128, callbacks=[es, rl, cp])
    # model.fit_generator(train, epochs=500, validation_data=(val),
    #             callbacks=[es, rl, cp])

    model.eval()
    
sub['digit']=np.argmax(model.predict(pred), axis=1)
# sub['digit']=np.argmax(model.predict_generator(pred), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples.csv', index=False)