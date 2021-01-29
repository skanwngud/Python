import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_iris

datasets=load_iris()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=33)

kf=KFold(n_splits=5, shuffle=True, random_state=24)

allAlgorithms=all_estimators(type_filter='classifier') # 분류형 모델 전체
# all_estimators - sklearn 0.20.0 대에 최적화

for (name, algorithm) in allAlgorithms: # name : 분류모델의 이름, algorithm : 분류모델
    try:
        model=algorithm()

        score=cross_val_score(model, x_train, y_train, cv=kf) # 위에서 n_splits=5 로 했기 때문에 cv=5 라고 해도 된다
        print(name, '의 정답률 : \n', score)
    except: # 예외 발생시 실행 할 행동
        print(name, '은 없는 모델') # continue 를 쓰면 무시하고 루프를 계속 돌게 됨


'''
0.23.2 에서 쓸 수 있는 모델들

tensorflow acc : 0.9666666388511658

AdaBoostClassifier 의 정답률 :  [0.91666667 0.95833333 0.95833333 0.91666667 0.91666667]
BaggingClassifier 의 정답률 :  [1.         0.95833333 0.95833333 0.91666667 0.91666667]
BernoulliNB 의 정답률 :  [0.25       0.25       0.33333333 0.25       0.25      ]
CalibratedClassifierCV 의 정답률 :  [1.         0.95833333 0.91666667 0.75       0.875     ]
CategoricalNB 의 정답률 :  [0.91666667 0.875      1.         0.91666667 0.91666667]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  [0.75       0.75       0.66666667 0.58333333 0.5       ]
DecisionTreeClassifier 의 정답률 :  [1.         0.95833333 0.91666667 0.95833333 0.91666667]
DummyClassifier 의 정답률 :  [0.33333333 0.29166667 0.25       0.41666667 0.29166667]
ExtraTreeClassifier 의 정답률 :  [1.         0.91666667 0.91666667 0.95833333 0.91666667]
ExtraTreesClassifier 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
GaussianNB 의 정답률 :  [1.         0.95833333 0.95833333 0.95833333 0.95833333]
GaussianProcessClassifier 의 정답률 :  [1.         0.95833333 0.95833333 1.         0.91666667]
GradientBoostingClassifier 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.91666667 0.91666667]
HistGradientBoostingClassifier 의 정답률 :  [1.         0.95833333 0.95833333 0.91666667 0.91666667]
KNeighborsClassifier 의 정답률 :  [1.         1.         1.         1.         0.91666667]
LabelPropagation 의 정답률 :  [0.95833333 1.         0.95833333 1.         0.91666667]
LabelSpreading 의 정답률 :  [0.95833333 1.         0.95833333 1.         0.91666667]
LinearDiscriminantAnalysis 의 정답률 :  [1.         1.         0.95833333 0.95833333 0.95833333]
LinearSVC 의 정답률 :  [1.         1.         0.95833333 0.91666667 0.91666667]
LogisticRegression 의 정답률 :  [1.         0.95833333 0.95833333 0.95833333 0.91666667]
LogisticRegressionCV 의 정답률 :  [1.         1.         0.95833333 0.95833333 0.91666667]
MLPClassifier 의 정답률 :  [1.         1.         0.95833333 0.95833333 0.91666667]
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  [0.625      0.875      0.66666667 0.66666667 0.875     ]
NearestCentroid 의 정답률 :  [0.875      0.91666667 0.95833333 1.         0.875     ]
NuSVC 의 정답률 :  [1.         0.91666667 0.95833333 0.95833333 0.91666667]
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  [0.95833333 0.75       0.66666667 0.66666667 0.875     ]
Perceptron 의 정답률 :  [1.         0.75       0.83333333 0.625      0.54166667]
QuadraticDiscriminantAnalysis 의 정답률 :  [1.         1.         0.95833333 0.95833333 0.95833333]
RadiusNeighborsClassifier 의 정답률 :  [0.91666667 0.91666667 0.95833333 1.         0.91666667]
RandomForestClassifier 의 정답률 :  [1.         0.95833333 0.95833333 0.95833333 0.91666667]
RidgeClassifier 의 정답률 :  [0.91666667 0.875      0.91666667 0.70833333 0.75      ]
RidgeClassifierCV 의 정답률 :  [0.91666667 0.875      0.91666667 0.70833333 0.75      ]
SGDClassifier 의 정답률 :  [0.95833333 0.91666667 0.70833333 0.75       0.5       ]
SVC 의 정답률 :  [1.         0.91666667 0.95833333 0.95833333 0.91666667]
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''