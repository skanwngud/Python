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

df_train=x_train
df_test=x_test

df_train=df_train.iloc[:, 2:]
df_test=df_test.iloc[:, 3:]

df_train['target1']=df_train.iloc[:, -1]
df_train['target2']=df_train.iloc[:, -1]

df_train.to_csv('./dacon/train/train_1.csv')
df_test.to_csv('./dacon/test/test_1.csv')

# print(df_train.shape) # (52560, 6)
# print(df_test.shape) # (27216, 6)

print(df_train.info())
# print(df_test.info())

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

df_train=df_train.reshape(-1, 48, 8)
df_test=df_test.reshape(-1, 7, 48, 6)

def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 2)

# print(x.shape) # (1087, 7, 48, 8)
# print(y.shape) # (1087, 2, 48, 8)
# print(df_test.shape) # (81, 7, 48, 6)

print(x[0])
