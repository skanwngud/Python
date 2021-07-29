parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsampe_bylevel':[0.6, 0.7, 0.9]}
]

# 0.95
# randomize, grid

import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

from tensorflow.keras.datasets import mnist

from xgboost import XGBClassifier

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

search=[GridSearchCV, RandomizedSearchCV]

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(-1, 28*28)
x_test=x_test.reshape(-1, 28*28)

pca=PCA(154)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

kf=KFold(n_splits=5, shuffle=True, random_state=23)

for i in search:
    model=i(XGBClassifier(n_jobs=8), parameters, cv=kf) # n_estimators == epochs

    model.fit(x_train, y_train, verbose=True,
        eval_metric='mlogloss', eval_set=[(x_train, y_train), (x_test, y_test)])
            # eval_metric = 'error' - accuracy
            # eval_metric = 'logloss' - loss
            # eval_set = validation
    acc=model.score(x_test, y_test)

    y_pred=model.predict(x_test)
    print(str(i)+' acc : ', acc)
    print(str(i)+' y_pred : ', np.argmax(y_pred[:10], axis=-1))
    print(str(i)+' y_test : ', np.argmax(y_test[:10], axis=-1))