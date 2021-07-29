# 1.0 xgb# 0.95 xgb

import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

# 1. data
(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(-1, 28*28)/255.
x_test=x_test.reshape(-1, 28*28)/255.

pca=PCA(n_components=712)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

# 2. model
model=XGBClassifier(n_jobs=8, use_label_encoder=False)

# 3. fitting
es=EarlyStopping(patience=10)
rl=ReduceLROnPlateau(verbose=1)

model.fit(x_train, y_train, eval_metric='logloss')
acc=model.score(x_test, y_test)

# 4. predict
y_pred=model.pred(x_test)

print(acc)
print(np.argmax(y_pred[:10], axis=-1))
print(np.argmax(y_test[:10], axis=-1))

# results
