import numpy as np

from sklearn.datasets import load_wine

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data
datasets=load_wine()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8, random_state=22)

# model
model=XGBClassifier(n_estimators=500, n_jobs=8, learning_rate=0.01)

# fitting
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric='mlogloss', verbose=1)

score=model.score(x_test, y_test)

y_pred=model.predict(x_test)

print('score : ', score)
print('acc : ', accuracy_score(y_test, y_pred))

# results
# score :  0.9166666666666666
# acc :  0.9166666666666666