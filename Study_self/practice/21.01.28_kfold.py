import warnings
warnings.filterwarnings('ignore')

from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.datasets import load_iris

datasets=load_iris()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=33)

kf=KFold(n_splits=5, shuffle=True, random_state=12)

allAlgorithms=all_estimators(type_filter='classifier')

for (name, algoritm) in allAlgorithms:
    try:
        model=algoritm()

        score=cross_val_score(model, x_train, y_train, cv=kf)
        # kf=KFold(n_splits=5) 라고 정의했으므로 cv=5 라고해도 된다
        # cv=5 로 하면 shuffle 이 되지 않아 가급적이면 kf 를 넣어주는 게 좋음
        print('score : ', score)
    except:
        continue
