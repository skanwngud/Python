# 가중치 저장
# model.save(), pickle

# 0. import labraries
import numpy as np
import tensorflow

from keras.models import Sequential, Model, save_model
from keras.layers import Dense, Dropout, Input
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# 1. data/preprocessing
from keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test) # 0 ~ 9

x_train=x_train.reshape(-1, 28*28).astype('float32')/255.
x_test=x_test.reshape(-1, 28*28).astype('float32')/255.

# 2. model
def build_model(drop=0.5, optimizer='adam'):
    inputs=Input(shape=(28*28), name='input') # 충돌 방지를 위한 name 으로 레이어 이름을 정해줌
    x=Dense(512, activation='relu', name='hidden1')(inputs)
    x=Dropout(drop)(x) # Dropout(0.5)
    x=Dense(256, activation='relu', name='hidden2')(x)
    x=Dropout(drop)(x)
    x=Dense(128, activation='relu', name='hidden3')(x)
    x=Dropout(drop)(x)
    outputs=Dense(10, activation='softmax', name='outputs')(x)
    model=Model(inputs, outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches=[10, 20, 30, 40, 50]
    optimizer=['rmsprop', 'adam', 'adadelta']
    dropout=[0.1, 0.2, 0.3]
    return {'batch_size' :  batches, 'optimizer' : optimizer,
            'drop' : dropout}

hyperparameters=create_hyperparameter()
model2=build_model()

from keras.wrappers.scikit_learn import KerasClassifier # sklearn 으로 싸겠단 의미

model2=KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search=RandomizedSearchCV(model2, hyperparameters, cv=3)
# search=GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)

search.best_estimator_.model.save('../data/h5/keras64_save_model2.h5')
# 그냥 model.save 하면 안 되고 best_estimator_ 를 사용해야 한다.

print(search.best_params_) # 내가 선택한 파라미터 중 가장 좋은 것
print(search.best_estimator_) # 전체 파라미터 중 가장 좋은 것
print(search.best_score_)
acc=search.score(x_test, y_test)

print('최종스코어 : ', acc)

# results

# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000281C5B131F0>        
# 0.9582499861717224
# 최종스코어 :  0.961899995803833

# pickle 은 먹히지 않음
# import pickle
# pickle.dump(search.best_estimator_, open('../data/h5/keras64_pickle2.dat', 'wb'))
