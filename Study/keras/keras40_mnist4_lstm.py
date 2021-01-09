# lstm model (N, 28, 28) -> (N, 28, 28, 1) -> (N, 764, 1) = (N, 28*14, 2) = (N, 28*28, 1)
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
