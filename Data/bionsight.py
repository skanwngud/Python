import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("cmpd.csv")

print(df.describe())

# train, test DataFrame 생성
train_set = pd.DataFrame(columns=['inchikey', 'smiles', 'activity'])
test_set = pd.DataFrame(columns=['inchikey', 'smiles', 'activity'])

for idx in range(len(df)):
    if df.iloc[idx, 2] == 'train':
        train_set.loc[idx] = df.iloc[idx, [0, 1, 3]].values
    else:
        test_set.loc[idx] = df.iloc[idx, [0, 1, 3]].values

print(train_set)
print(test_set)