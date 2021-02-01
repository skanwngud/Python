import numpy as np
import pandas as pd

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA

# print(train.info()) # (2048, 786)
# print(pred.info()) # (20480, 784)

x=train.iloc[:, 2:]
y=train.iloc[:, 0]

x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

# print(x.shape) # (2048, 785)
# print(y.shape) # (2048, )
# print(pred.shape) # (2048, 785)

kf=KFold(n_splits=5, shuffle=True, random_state=12)

for train_index, test_index in kf.split(x,y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

# x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=99)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=99)

pca=PCA(94)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
x_val=pca.transform(x_val)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

# print(np.argmax(np.cumsum(pca.explained_variance_ratio_)>=0.95)+1) # 94

print(x_train.shape) # (1311, 94)
print(x_test.shape) # (409, 94)
print(x_val.shape) # (328, 94)
print(y_train.shape) # (1311, )
print(y_test.shape) # (409, )
print(y_val.shape) # (328, )

model=Sequential()
model.add(Conv1D(128, 2, padding='same', activation='relu',input_shape=(94, )))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same', activation='relu'))
model.add(Conv1D(512, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same', activation='relu'))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Conv1D(128, padding='same', activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=100, batch_size=64, )