import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_wine

datasets=load_wine()
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
AdaBoostClassifier 의 정답률 : 
 [0.79310345 0.86206897 0.32142857 0.96428571 0.92857143]
BaggingClassifier 의 정답률 : 
 [0.93103448 0.96551724 0.89285714 0.96428571 1.        ]
BernoulliNB 의 정답률 :
 [0.37931034 0.24137931 0.32142857 0.42857143 0.35714286]
CalibratedClassifierCV 의 정답률 : 
 [0.89655172 0.89655172 0.85714286 0.92857143 0.96428571]
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :
 [0.65517241 0.68965517 0.57142857 0.67857143 0.71428571]
DecisionTreeClassifier 의 정답률 : 
 [0.89655172 0.86206897 0.96428571 0.92857143 1.        ]
DummyClassifier 의 정답률 :
 [0.34482759 0.34482759 0.35714286 0.32142857 0.46428571]
ExtraTreeClassifier 의 정답률 : 
 [0.82758621 0.93103448 0.89285714 0.78571429 0.92857143]
ExtraTreesClassifier 의 정답률 : 
 [0.96551724 0.96551724 1.         1.         1.        ]
GaussianNB 의 정답률 :
 [0.93103448 1.         1.         1.         1.        ]
GaussianProcessClassifier 의 정답률 : 
 [0.27586207 0.31034483 0.57142857 0.32142857 0.42857143]
GradientBoostingClassifier 의 정답률 : 
 [0.93103448 0.93103448 0.96428571 0.82142857 1.        ]
HistGradientBoostingClassifier 의 정답률 : 
 [0.96551724 0.89655172 0.96428571 1.         1.        ]
KNeighborsClassifier 의 정답률 :
 [0.75862069 0.72413793 0.78571429 0.82142857 0.78571429]
LabelPropagation 의 정답률 : 
 [0.51724138 0.44827586 0.46428571 0.5        0.5       ]
LabelSpreading 의 정답률 :
 [0.51724138 0.44827586 0.46428571 0.5        0.5       ]
LinearDiscriminantAnalysis 의 정답률 : 
 [0.96551724 1.         1.         1.         1.        ]
LinearSVC 의 정답률 : 
 [0.86206897 0.72413793 0.92857143 0.89285714 0.96428571]
LogisticRegression 의 정답률 : 
 [0.93103448 0.93103448 0.96428571 0.92857143 1.        ]
LogisticRegressionCV 의 정답률 : 
 [0.93103448 0.93103448 0.96428571 1.         1.        ]
MLPClassifier 의 정답률 : 
 [0.65517241 0.24137931 0.14285714 0.96428571 0.39285714]
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :
 [0.75862069 0.89655172 0.82142857 0.92857143 1.        ]
NearestCentroid 의 정답률 :
 [0.62068966 0.62068966 0.67857143 0.71428571 0.92857143]
NuSVC 의 정답률 :
 [0.82758621 0.89655172 0.89285714 0.85714286 1.        ]
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 : 
 [0.65517241 0.55172414 0.57142857 0.64285714 0.67857143]
Perceptron 의 정답률 : 
 [0.68965517 0.65517241 0.57142857 0.57142857 0.67857143]
QuadraticDiscriminantAnalysis 의 정답률 :
 [1.         1.         1.         0.96428571 1.        ]
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 : 
 [0.96551724 0.93103448 1.         1.         1.        ]
RidgeClassifier 의 정답률 :
 [0.93103448 1.         1.         1.         1.        ]
RidgeClassifierCV 의 정답률 : 
 [0.93103448 1.         1.         1.         1.        ]
SGDClassifier 의 정답률 : 
 [0.65517241 0.62068966 0.57142857 0.60714286 0.35714286]
SVC 의 정답률 :
 [0.72413793 0.65517241 0.60714286 0.71428571 0.67857143]
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''