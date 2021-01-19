import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train=pd.read_csv('./dacon/train/train.csv', header=0, index_col=0)
sub=pd.read_csv('./dacon/sample_submission.csv', header=0, index_col=0)

df_test = []

for i in range(81):
    file_path = './dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    df_test.append(temp)

x_test = pd.concat(df_test)

# print(x_train.info())
# print(x_test.info())

df_train=x_train.iloc[:, 2:]
df_test=x_test.iloc[:, 3:]

# print(df_train.info())

df_test.to_csv('./dacon/dacon_test.csv', sep=',')

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

# print(df_train.shape) # (52560, 6)

df_train=df_train.reshape(-1, 48, 6)
df_test=df_test.reshape(-1, 7, 48, 6)

def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col-1
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 3)