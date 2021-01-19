import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

train=pd.read_csv('./dacon/train/train.csv')
sub=pd.read_csv('./dacon/sample_submission.csv')

df_test=list()

for i in range(81):
    file_path='./dacon/test/' +str(i)+ '.csv'
    temp=pd.read_csv(file_path)
    df_test.append(temp)

df_x_test=pd.concat(df_test)

print(df_x_test.shape)

df_x_test.to_csv('./dacon/df_x_test_baseline.csv')

df_train=train.drop(['Day','Hour', 'Minute'], axis=1)
df_test=df_x_test.drop(['Day', 'Hour', 'Minute'], axis=1)

print(df_train.info())
print(df_test.info())