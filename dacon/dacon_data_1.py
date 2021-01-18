import numpy as np
import pandas as pd
import os
import glob


x_train=pd.read_csv('./dacon/train/train.csv', header=0, index_col=0)
sub=pd.read_csv('./dacon/sample_submission.csv', header=0, index_col=0)


def preprocess_data(data):
        temp = data.copy()
        return temp.iloc[-48:,:]

df_test = []

for i in range(81):
    file_path = './dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    temp=preprocess_data(temp)
    df_test.append(temp)

x_test = pd.concat(df_test)
x_test = x_test.append(x_test[-96:])

print(x_train.shape) # (52560, 8)
print(x_test.shape)