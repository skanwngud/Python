# pipeline 위주로 지금까지 배운 것 총동원

import tensorflow
import numpy as np
import pandas as pd

from keras.datasets import mnist

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold, train_test_split,\
                        GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# data
(x_train, y_train), (x_test, y_test)=mnist.load_data()

kf=KFold(n_splits=5, shuffle=True, random_state=23)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

x_train=x_train.reshape(-1, 28*28)/255.
x_test=x_test.reshape(-1, 28*28)/255.

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=23)

# model
def build_model(optimizer='adam', drop=0.2):
    optimizer=optimizer
    inputs=Input(shape=(28*28), name='input')
    x=Dense(64, name='hidden1')(inputs)
    x=BatchNormalization(name='bn1')(x)
    x=Activation('relu', name='act1')(x)
    x=Dropout(drop, name='drop1')(x)
    x=Dense(128, name='hidden2')(x)
    x=BatchNormalization(name='bn2')(x)
    x=Activation('relu', name='act2')(x)
    x=Dropout(drop, name='drop2')(x)
    x=Dense(256, name='hidden3')(x)
    x=BatchNormalization(name='bn3')(x)
    x=Activation('relu', name='act3')(x)
    x=Dropout(drop, name='drop3')(x)
    x=Dense(256, name='hidden4')(x)
    x=BatchNormalization(name='bn4')(x)
    x=Activation('relu', name='act4')(x)
    x=Dropout(drop, name='drop4')(x)
    outputs=Dense(10, activation='softmax', name='output')(x)
    model=Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, metrics=['acc'])
    return model

def creat_parameter():
    batches=[16, 32, 64, 128]
    optimizers=['rmsprop', 'adam', 'adadelta']
    dropout=[0.1, 0.2, 0.3, 0.4]
    return {'optimizer' : optimizers, 'drop' : dropout, 'batch_size' : batches}

model2=build_model()
hyperparameter=creat_parameter()

es=EarlyStopping(patience=50, verbose=1)
rl=ReduceLROnPlateau(patience=10, verbose=1)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/keras65_pipeline.hdf5',
                    save_best_only=True, verbose=1)

model2=KerasClassifier(build_fn=build_model(), verbose=1)
pipe=make_pipeline(MinMaxScaler(), model2, verbose=1)

search=RandomizedSearchCV(pipe, hyperparameter, cv=kf, verbose=1)


search.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
            callbacks=[es, cp, rl])

acc=search.score(x_test, y_test)

print('best estimator : ', search.best_estimator_)
print('best params : ', search.best_params_)
print('best score : ', search.best_score_)
