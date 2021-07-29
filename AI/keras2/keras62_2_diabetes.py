# DNN
# import labraries
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# data
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

print(x_train.shape) # (353, 10)
print(y_train.shape) # (353, )


# model
def build_model(activation='relu', batches=64, learning_rate=0.01,
                optimizer='adam'):
    model=Sequential()
    model.add(Dense(64, activation=activation, input_shape=(10, )))
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
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/keras62_diabetes_{val_loss:.4f}.hdf5',
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
print(search.cv_results_)

print('최종 스코어 : ', mse)

'''
results

<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x00000259D1A60F40>
{'optimizer': 'adam', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'relu'}
-2987.4339192708335
{'mean_fit_time': array([4.83747602, 2.15080174, 2.20994671, 3.51735894, 2.54875771,
       3.30102952, 1.74655477, 3.34520507, 2.48147655, 3.14962141]), 'std_fit_time': array([2.37179896, 0.14880233, 0.12712615, 0.83248613, 0.29909445,
       0.02812029, 0.12752495, 0.05010478, 0.24957898, 0.04610994]), 'mean_score_time': array([0.03315115, 0.04019737, 0.04384939, 0.04366986, 0.0331947 ,
       0.03208756, 0.03362203, 0.03392148, 0.03394063, 0.03508766]), 'std_score_time': array([0.00054809, 0.00819601, 0.00085599, 0.00071038, 0.00080224,
       0.00061455, 0.00050509, 0.00140189, 0.00082466, 0.00012776]), 'param_optimizer': masked_array(data=['adadelta', 'adam', 'adam', 'rmsprop', 'rmsprop',
                   'adadelta', 'adam', 'adadelta', 'rmsprop', 'adadelta'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_learning_rate': masked_array(data=[0.01, 0.01, 0.001, 0.01, 0.1, 0.001, 0.001, 0.01, 0.01,
                   0.01],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_batch_size': masked_array(data=[64, 64, 16, 16, 64, 128, 64, 128, 64, 32],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'param_activation': masked_array(data=['linear', 'relu', 'relu', 'relu', 'linear', 'relu',
                   'linear', 'linear', 'linear', 'linear'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False],
       fill_value='?',
            dtype=object), 'params': [{'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'adam', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'relu'}, {'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 16, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.01, 'batch_size': 16, 'activation': 'relu'}, {'optimizer': 'rmsprop', 'learning_rate': 0.1, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.001, 'batch_size': 128, 'activation': 'relu'}, {'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 128, 'activation': 'linear'}, {'optimizer': 'rmsprop', 'learning_rate': 0.01, 'batch_size': 64, 'activation': 'linear'}, {'optimizer': 'adadelta', 'learning_rate': 0.01, 'batch_size': 32, 'activation': 'linear'}], 'split0_test_score': array([-2936.32373047, -2472.10766602, -2572.86865234, -2575.08154297,
       -2773.60083008, -2531.13427734, -2785.21044922, -2916.48413086,
       -2729.12792969, -2982.58056641]), 'split1_test_score': array([ -3990.59155273,  -3388.82763672,  -3308.60839844,  -3191.90380859,
        -3390.65136719, -14079.62402344,  -3326.85351562,  -3656.61914062,
        -3320.72192383,  -3382.35766602]), 'split2_test_score': array([-4294.82763672, -3101.36645508, -3135.63696289, -3213.14575195,
       -3066.23974609, -3668.14477539, -3024.65063477, -3893.06689453,
       -3075.29223633, -3633.99145508]), 'mean_test_score': array([-3740.58097331, -2987.43391927, -3005.70467122, -2993.37703451,
       -3076.83064779, -6759.63435872, -3045.5715332 , -3488.72338867,
       -3041.71402995, -3332.9765625 ]), 'std_test_score': array([ 582.10089801,  382.82227661,  314.10194079,  295.90667867,
        252.02111914, 5196.78649018,  221.61914153,  415.98895115,
        242.68152716,  268.21994272]), 'rank_test_score': array([ 9,  1,  3,  2,  6, 10,  5,  8,  4,  7])}
최종 스코어 :  -3245.010498046875
'''