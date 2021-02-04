import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow

from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

from sklearn.model_selection import train_test_split

import cv2

import gc
from keras import backend as bek

train=pd.read_csv('../data/dacon/data/train.csv')
test=pd.read_csv('../data/dacon/data/test.csv')

x_train=train.drop(['id', 'letter', 'digit'], axis=1).values
x_train=x_train.reshape(-1, 28, 28, 1)

x_train=np.where((x_train<=20)&(x_train!=0), 0., x_train)

x_train=x_train/255
x_train=x_train.astype('float32')

y=train['digit']
y_train=np.zeros((len(y), len(y.unique()))) # 0 ~ 9

for i, digit in enumerate(y): # y 안의 i 와 digit
    y_train[i, digit]=1

train_224=np.zeros([2048, 56, 56, 3], dtype=np.float32)

for i,s in enumerate(x_train):
    converted=cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized=cv2.resize(converted, (56, 56), interpolation=cv2.INTER_CUBIC)
    del converted
    train_224[i]=resized
    del resized
    bek.clear_session()
    gc.collect()

datagen=ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range=10,
        validation_split=0.2
)

valgen=ImageDataGenerator()

from keras.callbacks import LearningRateScheduler, EarlyStopping

def create_model():
    effnet=tf.keras.applications.EfficientNetB3(
        include_top=True,
        weights=None,
        input_shape=(56, 56, 3),
        classes=10,
        classifier_activation='softmax',
    )

    model=Sequential()
    model.add(effnet)
    
    model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=initial_learningrate),
                    metrics=['acc'])
    return model

initial_learningrate=2e-3

from sklearn.model_selection import RepeatedKFold

kf=RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores=[]
Fold=1
results=np.zeros((20480, 10))

def lr_decay(epoch):
    return initial_learningrate*0.99**epoch

x_test=test.drop(['id', 'letter'], axis=1).values
x_test=x_test.reshape(-1, 28, 28, 1)
x_test=np.where((x_test<=20)&(x_test!=0), 0., x_test)
x_test=x_test/255
x_test=x_test.astype('float32')

test_224=np.zeros([20480, 56, 56, 3], dtype=np.float32)

for i, s in enumerate(x_test):
    converted=cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized=cv2.resize(converted, (56, 56), interpolation=cv2.INTER_CUBIC)
    del converted
    test_224[i]=resized
    del resized

bek.clear_session()
gc.collect()

results=np.zeros((20480, 10), dtype=np.float32)

for train, val in kf.split(train_224):
    initial_learningrate=2e-3
    es=EarlyStopping(patience=50)
    filepath_val_acc='../data/modelcheckpoint/effi_model_aug'+str(Fold)+'.ckpt'
    checkpoint_val_acc=ModelCheckpoint(filepath_val_acc, monitor='val_acc',
                    verbose=1, save_best_only=True, save_weights_only=True)
    gc.collect()
    bek.clear_session()

    print('Fold : ', Fold)
    
    x_train=train_224[train]
    x_val=train_224[val]
    x_train=x_train.astype('float32')
    x_val=x_val.astype('float32')

    y_train=y_train[train]
    y_val=y_train[val]

    model=create_model()

    training_generator=datagen.flow(x_train, y_train, batch_size=4, seed=7, shuffle=True)
    validation_generator=valgen.flow(x_val, y_val, batch_size=4, seed=7, shuffle=True)
    model.fit(training_generator, epochs=150, callbacks=[LearningRateScheduler(lr_decay), es, checkpoint_val_acc],
                shuffle=True,
                validation_data=validation_generator,
                steps_per_epoch=len(x_train)/32)

    del x_train
    del x_val
    del y_train
    del y_val

    gc.collect()
    bek.clear_session()
    model.load_weights(filepath_val_acc)
    results=results+model.predict(test_224)

    Fold=Fold+1

submission=pd.read_csv('../data/dacon/data/sample_submission.csv')
submission['digit']=np.argmax(results, axis=1)

submission.head()
submission.to_csv('../data/dacon/data/private2.csv', index=False)