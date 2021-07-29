# DNN
# import labraries
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.datasets import load_wine
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# data
datasets=load_wine()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

print(x_train.shape) # (142, 13)
print(y_train.shape) # (142, )

# model
def build_model(activation='relu', batches=64, learning_rate=0.01,
                optimizer='adam'):
    model=Sequential()
    model.add(Dense(64, activation=activation, input_shape=(13, )))
    model.add(Dense(128, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def create_parameter():
    batches=[16, 32, 64, 128]
    optimizer=['rmsprop', 'adam', 'adadelta']
    activation=['relu','linear']
    learning_rate=[0.1, 0.01, 0.001]
    return {'batch_size' :  batches, 'optimizer' : optimizer,
            'activation' : activation, 'learning_rate':learning_rate}

es=EarlyStopping(patience=30, verbose=1)
rl=ReduceLROnPlateau(patience=5, verbose=1)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/keras62_wine_{val_loss:.4f}.hdf5',
                    verbose=1, save_best_only=True)

model2=build_model()
hyperparameters=create_parameter()

model2=KerasClassifier(build_fn=build_model, verbose=1, epochs=100)
search=RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, validation_split=0.2,
            epochs=100, batch_size=64, callbacks=[es, rl, cp])

mse=search.score(x_test, y_test)

print(search.best_estimator_)
print(search.best_params_)
print(search.best_score_)
print(search.cv_results_)

print('최종 스코어 : ', mse)

'''
results

<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F5F9459CA0>
{'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'linear'}
0.7467494010925293
{'mean_fit_time': array([2.6527497 , 2.38648017, 2.73049911, 2.98740959, 3.10056718,
       2.85677338, 2.925994  , 2.5865074 , 1.81144945, 1.63133907]), 'std_fit_time': array([0.73194155, 0.05466445, 0.28939548, 0.72883987, 0.13800516,
       0.58955716, 0.12164671, 0.25418172, 0.15650839, 0.23823475]), 'mean_score_time': array([0.03513924, 0.03139806, 0.03219732, 0.03457435, 0.0313344 ,
       0.03104599, 0.031051  , 0.03524121, 0.03105704, 0.0637087 ]), 'std_score_time': array([0.00125195, 0.0004081 , 0.00041835, 0.00124396, 0.00042321,
       0.00049013, 0.00023496, 0.00495233, 0.00017569, 0.04169388]), 'param_optimizer': masked_array(data=['adam', 'rmsprop', 'rmsprop', 'rmsprop', 'adadelta',
                   'rmsprop', 'adadelta', 'rmsprop', 'adam', 'adam'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_learning_rate': masked_array(data=[0.01, 0.001, 0.1, 0.01, 0.001, 0.001, 0.1, 0.01, 0.01,
                   0.1],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[32, 128, 128, 32, 128, 64, 64, 64, 128, 16],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_activation': masked_array(data=['relu', 'linear', 'linear', 'relu', 'relu', 'relu',
                   'relu', 'linear', 'relu', 'linear'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'adam', 'learning_rate': 0.01, 'batch_size': 32, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.1, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.01, 'batch_size': 32, 'activation': 'relu'}, {'optimizer': 'adadelta', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 64, 'activation': 'relu'}, {'optimizer': 'adadelta', 'learning_rate': 0.1, 'batch_size': 64, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'adam', 'learning_rate': 0.01, 'batch_size': 128, 'activation': 'relu'}, {'optimizer': 'adam', 'learning_rate': 0.1, 'batch_size': 16, 'activation': 'linear'}], 'split0_test_score': array([0.66666669, 0.70833331, 0.77083331, 0.64583331, 0.64583331,
       0.64583331, 0.60416669, 0.60416669, 0.60416669, 0.64583331]), 'split1_test_score': array([0.68085104, 0.82978725, 0.68085104, 0.65957445, 0.61702126,
       0.57446808, 0.61702126, 0.76595747, 0.59574467, 0.63829786]), 'split2_test_score': array([0.68085104, 0.70212764, 0.68085104, 0.57446808, 0.68085104,
       0.72340423, 0.68085104, 0.78723407, 0.65957445, 0.59574467]), 'mean_test_score': array([0.67612292, 0.7467494 , 0.71084513, 0.62662528, 0.64790187,
       0.64790187, 0.634013  , 0.71911941, 0.6198286 , 0.62662528]), 'std_test_score': array([0.00668657, 0.05877126, 0.04241805, 0.03730492, 0.02609942,
       0.06082052, 0.03353269, 0.08174664, 0.02831409, 0.02205153]), 'rank_test_score': array([ 4,  1,  3,  8,  5,  5,  7,  2, 10,  8])}
최종 스코어 :  0.6944444179534912

'''