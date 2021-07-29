# DNN
# import labraries
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor

from keras.datasets import boston_housing

from sklearn.model_selection import RandomizedSearchCV

# data
(x_train, y_train), (x_test, y_test)=boston_housing.load_data()

print(x_train.shape) # (404, 13)
print(y_train.shape) # (404, )

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
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
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
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/keras62_boston_{val_loss:.4f}.hdf5',
                    verbose=1, save_best_only=True)

model2=build_model()
hyperparameters=create_parameter()

model2=KerasRegressor(build_fn=build_model, verbose=1, epochs=100)
search=RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, validation_split=0.2,
            epochs=100, batch_size=64, callbacks=[es, rl, cp])

mse=search.score(x_test, y_test)

print(search.best_estimator_)
print(search.best_params_)
print(search.best_score_)

print('최종 스코어 : ', mse)

# results
# <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x0000026CA149F2E0>
# {'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'relu'}
# -59.750118255615234
# 최종 스코어 :  -62.9643440246582