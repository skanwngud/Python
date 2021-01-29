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
        print('score : ', score, ' - ', name)
    except:
        continue

'''
score :  [1.         0.875      1.         0.83333333 0.95833333]  -  AdaBoostClassifier
score :  [1.         0.875      1.         0.91666667 0.95833333]  -  BaggingClassifier
score :  [0.375      0.25       0.375      0.29166667 0.25      ]  -  BernoulliNB
score :  [1.         0.83333333 0.91666667 0.91666667 0.875     ]  -  CalibratedClassifierCV
score :  [1.         0.83333333 0.95833333 1.         0.875     ]  -  CategoricalNB
score :  [0. 0. 0. 0. 0.]  -  CheckingClassifier
score :  [0.625      0.75       0.625      0.70833333 0.54166667]  -  ComplementNB
score :  [1.         0.875      0.95833333 0.91666667 0.95833333]  -  DecisionTreeClassifier
score :  [0.25       0.33333333 0.41666667 0.375      0.375     ]  -  DummyClassifier
score :  [1.         0.83333333 0.91666667 0.95833333 0.91666667]  -  ExtraTreeClassifier
score :  [1.         0.875      1.         0.95833333 0.91666667]  -  ExtraTreesClassifier
score :  [1.         0.91666667 1.         0.95833333 0.95833333]  -  GaussianNB
score :  [1.         0.875      1.         0.95833333 1.        ]  -  GaussianProcessClassifier
score :  [1.         0.875      1.         0.91666667 0.95833333]  -  GradientBoostingClassifier
score :  [1.         0.875      1.         0.91666667 0.95833333]  -  HistGradientBoostingClassifier
score :  [1.         0.91666667 1.         1.         0.95833333]  -  KNeighborsClassifier
score :  [1.         0.875      1.         0.95833333 0.91666667]  -  LabelPropagation
score :  [1.         0.875      1.         0.95833333 0.91666667]  -  LabelSpreading
score :  [1.         0.95833333 1.         0.95833333 0.95833333]  -  LinearDiscriminantAnalysis
score :  [1.         0.95833333 1.         0.91666667 0.95833333]  -  LinearSVC
score :  [1.         0.875      1.         0.95833333 0.95833333]  -  LogisticRegression
score :  [1.         0.875      1.         0.95833333 0.95833333]  -  LogisticRegressionCV
score :  [1.         0.95833333 1.         0.91666667 0.91666667]  -  MLPClassifier
score :  [0.83333333 0.625      0.875      0.70833333 1.        ]  -  MultinomialNB
score :  [0.95833333 0.83333333 0.95833333 0.95833333 0.875     ]  -  NearestCentroid
score :  [1.         0.875      1.         0.95833333 1.        ]  -  NuSVC
score :  [0.95833333 0.79166667 0.95833333 0.875      0.58333333]  -  PassiveAggressiveClassifier
score :  [0.79166667 0.83333333 0.66666667 0.5        0.54166667]  -  Perceptron
score :  [1.         0.95833333 1.         0.95833333 0.91666667]  -  QuadraticDiscriminantAnalysis
score :  [1.    0.875 1.    1.    1.   ]  -  RadiusNeighborsClassifier
score :  [1.         0.875      1.         0.91666667 0.95833333]  -  RandomForestClassifier
score :  [0.95833333 0.79166667 0.83333333 0.79166667 0.875     ]  -  RidgeClassifier
score :  [0.95833333 0.79166667 0.83333333 0.79166667 0.875     ]  -  RidgeClassifierCV
score :  [0.625      0.625      0.875      0.83333333 0.91666667]  -  SGDClassifier
score :  [1.         0.83333333 1.         0.95833333 1.        ]  -  SVC
'''