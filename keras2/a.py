import pickle
from tensorflow.keras.models import load_model

import tensorflow

from keras.models import Sequential
from keras.layers import Dense


model=pickle.load(open('../data/h5/keras64_pickle.dat', 'rb'))