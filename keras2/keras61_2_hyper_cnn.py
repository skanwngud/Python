# cnn, parameter

# 0. import labraries
import numpy as np
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# 1. data/preprocessing
from keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test) # 0 ~ 9

x_train=x_train.reshape(-1, 28, 28, 1).astype('float32')/255.
x_test=x_test.reshape(-1, 28, 28, 1).astype('float32')/255.

# 2. model
def build_model(drop=0.5, optimizer='adam', node=512, activation='relu', lr=0.1):
    optimizer=optimizer(lr=lr)
    inputs=Input(shape=(28, 28, 1), name='input') # 충돌 방지를 위한 name 으로 레이어 이름을 정해줌
    x=Conv2D(filters=node, kernel_size=2, activation=activation, name='hidden1')(inputs)
    x=Dropout(drop)(x) # Dropout(0.5)
    x=Conv2D(filters=node, kernel_size=2, activation=activation, name='hidden2')(x)
    x=Dropout(drop)(x)
    x=Conv2D(filters=node, kernel_size=2, activation=activation, name='hidden3')(x)
    x=Dropout(drop)(x)
    x=Flatten()(x)
    outputs=Dense(10, activation='softmax', name='outputs')(x)
    model=Model(inputs, outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches=[10, 20, 30, 40, 50, 60, 70]
    optimizer=['rmsprop', 'adam', 'adadelta']
    dropout=[0.1, 0.2, 0.3]
    lr=[0.1, 0.01, 0.001, 0.0001]
    nodes=[1024, 512]
    activation=['relu', 'linear']
    return {'batch_size' :  batches, 'optimizer' : optimizer, 'lr' : lr,
            'drop' : dropout, 'activation' : activation, 'node' : nodes}

hyperparameters=create_hyperparameter()
model2=build_model()

from keras.wrappers.scikit_learn import KerasClassifier # sklearn 으로 싸겠단 의미

model2=KerasClassifier(build_fn=build_model, verbose=1)
# 아까 정의해줬던 model 을 KerasClassfier 로 싸서 sklearn 이 인식하게끔 만들어줌

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search=RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # 내가 선택한 파라미터 중 가장 좋은 것
print(search.best_estimator_) # 전체 파라미터 중 가장 좋은 것
print(search.best_score_)
# best_params_, best_estimator_ 둘 중 하나만 먹힘

# TypeError: If no scoring is specified, the estimator passed should have a 'score' method.
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001C41E277DF0> does not.
# keras model 을 sklearn 과 엮으면 위에 해당하는 Type error 가 나온다.
# 해결하기 위해선 wrapping 을 해줘야한다.
acc=search.score(x_test, y_test)
print('최종스코어 : ', acc)

# results
# {'optimizer': 'rmsprop', 'node': 512, 'drop': 0.2, 'batch_size': 60, 'activation': 'relu'}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001C46BD0FCA0>
# 0.9768499930699667
# 167/167 [==============================] - 3s 17ms/step - loss: 0.0800 - acc: 0.9758
# 최종스코어 :  0.9757999777793884