import numpy as np
import pandas as pd

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)
pred=pd.read_csv('../data/dacon/data/test.csv', index_col=0, header=0)
sub=pd.read_csv('../data/dacon/data/sample_submission.csv')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# print(train.info()) # (2048, 786)
# print(pred.info()) # (20480, 784)

# data
x=train.iloc[:, 2:]
y=train.iloc[:, 0]
pred=pred.iloc[:, 1:]

x=x.to_numpy()
y=y.to_numpy()
pred=pred.to_numpy()

# print(x.shape) # (2048, 785)
# print(y.shape) # (2048, )
# print(pred.shape) # (2048, 785)

kf=KFold(n_splits=5, shuffle=True, random_state=12)

# for train_index, test_index in kf.split(x,y):
#     x_train=x[train_index]
#     x_test=x[test_index]
#     y_train=y[train_index]
#     y_test=y[test_index]

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=99)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=99)

pca=PCA(94)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
x_val=pca.transform(x_val)
pred=pca.transform(pred)

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

x_train=x_train.reshape(-1, 94, 1)
x_test=x_test.reshape(-1, 94, 1)
x_val=x_val.reshape(-1, 94, 1)

pred=pred.reshape(-1, 94, 1)

print(pred.shape) # (20480, 94)


# model
# model=XGBClassifier(n_estimators=1000, n_jobs=8, learning_rate=0.01)

# model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
#             eval_metric=['mlogloss'], early_stopping_rounds=30, verbose=1)

# score=model.score(x_test, y_test)

# print('score : ', score) # score :  0.2658536585365854
model=Sequential()
model.add(Conv1D(128, 2, padding='same', activation='relu',input_shape=(94, 1)))
model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(512, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(256, 3, padding='same', activation='relu'))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

es=EarlyStopping(patience=50)
rl=ReduceLROnPlateau(patience=10, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=1000, batch_size=64, callbacks=[es, rl])
# fitting

# model=KerasClassifier(build_fn=model, epochs=100, batch_size=64)
# cross_val_score(model, x_test, y_test, cv=kf)

# eval, pred
loss=model.evaluate(x_test, y_test)
# y_pred=model.predict(pred)

print(loss) # [7.086297988891602, 0.22439023852348328]

sub['digit']=np.argmax(model.predict(pred), axis=1)

print(sub.head())

sub.to_csv('../data/dacon/data/samples3.csv', index=False)