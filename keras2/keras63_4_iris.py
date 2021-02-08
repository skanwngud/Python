# DNN
# import labraries
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# data
datasets=load_iris()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

print(x_train.shape) # (120, 4)
print(y_train.shape) # (120, )


# model
def build_model(activation='relu', batches=64, learning_rate=0.01,
                optimizer='adam'):
    model=Sequential()
    model.add(Dense(64, activation=activation, input_shape=(4, )))
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
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/keras62_iris_{val_loss:.4f}.hdf5',
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

<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000127A05F44C0>
{'optimizer': 'adam', 'learning_rate': 0.1, 'batch_size': 64, 'activation': 'relu'}
0.9833333492279053
{'mean_fit_time': array([4.83248488, 1.86302884, 2.53405762, 1.68893456, 2.49936088,
       2.51732302, 2.13943617, 1.69613798, 2.80785441, 2.95535231]), 'std_fit_time': array([2.08133928, 0.55538639, 0.04666469, 0.11694567, 0.04119143,
       0.08803447, 0.10896907, 0.09283257, 0.53283462, 0.3111602 ]), 'mean_score_time': array([0.03368696, 0.03303925, 0.03315369, 0.03232066, 0.03279519,
       0.03426258, 0.03157942, 0.03455043, 0.0320762 , 0.03492578]), 'std_score_time': array([0.00119641, 0.00047862, 0.00179822, 0.00043306, 0.00067298,
       0.00045765, 0.00091497, 0.00045429, 0.00030811, 0.00075118]), 'param_optimizer': masked_array(data=['adadelta', 'adam', 'adadelta', 'adam', 'adadelta',
                   'adadelta', 'rmsprop', 'adam', 'rmsprop', 'rmsprop'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_learning_rate': masked_array(data=[0.1, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.001, 0.001,
                   0.001],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[32, 128, 64, 64, 32, 16, 128, 32, 64, 32],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_activation': masked_array(data=['linear', 'linear', 'linear', 'relu', 'linear',
                   'linear', 'relu', 'relu', 'linear', 'linear'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'adadelta', 'learning_rate': 0.1, 'batch_size': 32, 'activation': 'linear'}, {'optimizer': 'adam', 'learning_rate': 0.1, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'adam', 'learning_rate': 0.1, 'batch_size': 64, 'activation': 'relu'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 32, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.1, 'batch_size': 16, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.01, 'batch_size': 128, 'activation': 'relu'}, {'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 32, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 32, 'activation': 'linear'}], 'split0_test_score': array([0.64999998, 0.875     , 0.64999998, 0.97500002, 0.625     ,
       0.64999998, 0.625     , 0.97500002, 0.94999999, 0.97500002]), 'split1_test_score': array([0.89999998, 0.69999999, 0.64999998, 1.        , 0.82499999,
       0.75      , 0.67500001, 0.875     , 0.92500001, 0.97500002]), 'split2_test_score': array([0.77499998, 0.94999999, 0.92500001, 0.97500002, 0.72500002,
       0.94999999, 0.67500001, 0.92500001, 0.92500001, 0.94999999]), 'mean_test_score': array([0.77499998, 0.84166666, 0.74166665, 0.98333335, 0.725     ,
       0.78333332, 0.65833334, 0.92500001, 0.93333334, 0.96666668]), 'std_test_score': array([0.10206207, 0.10474838, 0.12963626, 0.0117851 , 0.08164965,
       0.12472192, 0.02357023, 0.04082484, 0.0117851 , 0.01178513]), 'rank_test_score': array([ 7,  5,  8,  1,  9,  6, 10,  4,  3,  2])}
최종 스코어 :  1.0
'''