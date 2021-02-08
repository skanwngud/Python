# DNN
# import labraries
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# data
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

print(x_train.shape) # (455, 30)
print(y_train.shape) # (455, )

# model
def build_model(activation='relu', batches=64, learning_rate=0.01,
                optimizer='adam'):
    model=Sequential()
    model.add(Dense(64, activation=activation, input_shape=(30, )))
    model.add(Dense(128, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def create_parameter():
    batches=[16, 32, 64, 128]
    optimizer=['rmsprop', 'adam', 'adadelta']
    activation=['relu', 'linear']
    learning_rate=[0.1, 0.01, 0.001]
    return {'batch_size' :  batches, 'optimizer' : optimizer,
            'activation' : activation, 'learning_rate':learning_rate}

es=EarlyStopping(patience=30, verbose=1)
rl=ReduceLROnPlateau(patience=5, verbose=1)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/keras62_cancer_{val_loss:.4f}.hdf5',
                    verbose=1, save_best_only=True)

model2=build_model()
hyperparameters=create_parameter()

model2=KerasClassifier(build_fn=build_model, verbose=1, epochs=100)
search=RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, validation_split=0.2,
            epochs=100, batch_size=64, callbacks=[es, rl, cp])

acc=search.score(x_test, y_test)

print(search.best_estimator_)
print(search.best_params_)
print(search.best_score_)
print(search.cv_results_)

print('최종 스코어 : ', acc)

'''
results

<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002CB1149E880>
{'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'relu'}
0.9209509690602621
{'mean_fit_time': array([5.10433269, 4.62021907, 2.90921513, 4.50235287, 2.9427913 ,
       2.96232104, 3.85776734, 2.20847313, 2.33900388, 3.57365243]), 'std_fit_time': array([0.73141341, 0.56496715, 0.75727161, 0.90087816, 0.19135238,
       0.09267233, 0.14222757, 0.36161489, 0.23289422, 0.08680662]), 'mean_score_time': array([0.04820426, 0.03457324, 0.04591425, 0.04682056, 0.03491489,
       0.03387014, 0.03463483, 0.03356258, 0.03957884, 0.03438449]), 'std_score_time': array([0.00094077, 0.00094061, 0.00081589, 0.00081825, 0.00029556,
       0.00081748, 0.00038354, 0.00024589, 0.00122057, 0.0014689 ]), 'param_optimizer': masked_array(data=['rmsprop', 'rmsprop', 'adadelta', 'rmsprop',
                   'adadelta', 'rmsprop', 'adadelta', 'adam', 'adadelta',
                   'adadelta'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_learning_rate': masked_array(data=[0.1, 0.001, 0.1, 0.01, 0.01, 0.001, 0.01, 0.01, 0.01,
                   0.001],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[16, 128, 16, 16, 128, 128, 64, 128, 32, 128],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_activation': masked_array(data=['relu', 'relu', 'linear', 'linear', 'linear', 'linear',
                   'relu', 'linear', 'linear', 'relu'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'rmsprop', 'learning_rate': 0.1, 'batch_size': 16, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'relu'}, {'optimizer': 'adadelta', 'learning_rate': 0.1, 'batch_size': 16, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.01, 'batch_size': 16, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'relu'}, {'optimizer': 'adam', 'learning_rate': 0.01, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 32, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'relu'}], 'split0_test_score': array([0.86842108, 0.8618421 , 0.80263156, 0.86842108, 0.81578946,
       0.85526317, 0.85526317, 0.86842108, 0.84210527, 0.88157892]), 'split1_test_score': array([0.93421054, 0.94736844, 0.84210527, 0.93421054, 0.91447371,
       0.93421054, 0.93421054, 0.93421054, 0.8881579 , 0.94736844]), 'split2_test_score': array([0.94701988, 0.95364237, 0.92052978, 0.91390729, 0.8874172 ,
       0.93377483, 0.94701988, 0.93377483, 0.71523178, 0.93377483]), 'mean_test_score': array([0.9165505 , 0.92095097, 0.85508887, 0.90551297, 0.87256012,
       0.90774951, 0.91216453, 0.91213548, 0.81516498, 0.9209074 ]), 'std_test_score': array([0.03443206, 0.04187469, 0.04899951, 0.02750651, 0.04163488,
       0.03711387, 0.04057374, 0.03091126, 0.0731218 , 0.02835775]), 'rank_test_score': array([ 3,  1,  9,  7,  8,  6,  4,  5, 10,  2])}
최종 스코어 :  0.9298245906829834
'''