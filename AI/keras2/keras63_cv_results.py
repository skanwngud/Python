# 61 copy
# model.cv_results

# 0. import labraries
import numpy as np
import tensorflow

from keras.models import Sequential, Model
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
# 아까 정의해줬던 model 을 KerasClassfier 로 싸서 sklearn 이 인식하게끔 만들어줌

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search=RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)

print(search.best_params_) # 내가 선택한 파라미터 중 가장 좋은 것
print(search.best_estimator_) # 전체 파라미터 중 가장 좋은 것
print(search.best_score_)
print(search.cv_results_)
# best_params_, best_estimator_ 둘 중 하나만 먹힘

# TypeError: If no scoring is specified, the estimator passed should have a 'score' method.
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001C41E277DF0> does not.
# keras model 을 sklearn 과 엮으면 위에 해당하는 Type error 가 나온다.
# 해결하기 위해선 wrapping 을 해줘야한다.
acc=search.score(x_test, y_test)
print('최종스코어 : ', acc)

'''
results

{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000024C54A09A60>
0.9554666876792908
{'mean_fit_time': array([ 5.07590024,  4.6074903 ,  3.13821212,  3.38062429, 13.07331459,
        3.42199238,  4.58181628,  8.89286153,  4.39520907,  4.38356336]), 'std_fit_time': array([0.30755139, 0.06280603, 0.04822569, 0.11503678, 0.33873331,
       0.05572256, 0.11539272, 0.08675135, 0.082882  , 0.03243678]), 'mean_score_time': array([1.2077318 , 1.32977955, 0.68088826, 1.02168695, 2.52471542,
       1.28507916, 1.30518365, 2.68312359, 1.24885408, 1.3284653 ]), 'std_score_time': array([0.24990637, 0.10979783, 0.14624176, 0.02485399, 0.05290294,
       0.07979403, 0.01319614, 0.36445677, 0.02843574, 0.09387153]), 'param_optimizer': masked_array(data=['rmsprop', 'adadelta', 'rmsprop', 'adadelta',
                   'rmsprop', 'adam', 'adam', 'adam', 'adadelta',
                   'adadelta'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.3, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[30, 20, 50, 30, 10, 30, 20, 10, 20, 20],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 10}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 20}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 20}], 'split0_test_score': array([0.95585001, 0.34009999, 0.95375001, 0.1728    , 0.95920002,
       0.95249999, 0.95880002, 0.94125003, 0.17135   , 0.37079999]), 'split1_test_score': array([0.95485002, 0.30450001, 0.95380002, 0.19984999, 0.95324999,
       0.95674998, 0.95235002, 0.92559999, 0.1992    , 0.38545001]), 'split2_test_score': array([0.95165002, 0.38080001, 0.95490003, 0.16635001, 0.94954997,
       0.95564997, 0.95525002, 0.94919997, 0.25764999, 0.36410001]), 'mean_test_score': array([0.95411668, 0.3418    , 0.95415002, 0.17966667, 0.954     ,
       0.95496664, 0.95546669, 0.93868333, 0.2094    , 0.37345   ]), 'std_test_score': array([0.00179133, 0.03117253, 0.00053073, 0.01451265, 0.00397515,
       0.00180107, 0.00263765, 0.00980411, 0.0359625 , 0.00891525]), 'rank_test_score': array([ 4,  8,  3, 10,  5,  2,  1,  6,  9,  7])}
500/500 [==============================] - 1s 1ms/step - loss: 0.1326 - acc: 0.9574
최종스코어 :  0.9574000239372253
'''