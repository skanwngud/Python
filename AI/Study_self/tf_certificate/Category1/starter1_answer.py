# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test model button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]

import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # YOUR CODE HERE
    # input=Input(shape=(1,))
    # dense=Dense(32)(dense)
    # batch=BatchNormalization()(dense)
    # act=Activation('relu')(batch)
    # dense=Dense(64)(dense)
    # batch=BatchNormalization()(dense)
    # act=Activation('relu')(batch)
    # dense=Dense(128)(dense)
    # batch=BatchNormalization()(dense)
    # act=Activation('relu')(batch)
    # output=Dense(1, activation='linear')(act)
    # model=Model(input, output)

    model=Sequential()
    model.add(Dense(128, activation='linear', input_dim=1))
    model.add(Dense(256, activation='linear'))
    model.add(Dense(512, activation='linear'))
    model.add(Dense(256, activation='linear'))
    model.add(Dense(128, activation='linear'))
    model.add(Dense(64, activation='linear'))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(16, activation='linear'))
    model.add(Dense(8, activation='linear'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics='mae'
    )    

    model.fit(
        xs, ys,
        epochs=200,
        batch_size=1
    )
    
    print(model.predict([10.0]))
    
    return model



# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
