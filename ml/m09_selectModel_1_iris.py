import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_iris

datasets=load_iris()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=33)

allAlgorithms=all_estimators(type_filter='classifier') # 분류형 모델 전체
# all_estimators - sklearn 0.20.0 대에 최적화
print(sklearn.__version__) # 0.23.2

for (name, algorithm) in allAlgorithms: # name : 분류모델의 이름, algorithm : 분류모델
    try:
        model=algorithm()

        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except: # 예외 발생시 실행 할 행동
        print(name, '은 없는 모델') # continue 를 쓰면 무시하고 루프를 계속 돌게 됨


'''
0.23.2 에서 쓸 수 있는 모델들

tensorflow acc : 0.9666666388511658

AdaBoostClassifier 의 정답률 :  0.9333333333333333
BaggingClassifier 의 정답률 :  0.9
BernoulliNB 의 정답률 :  0.26666666666666666
CalibratedClassifierCV 의 정답률 :  0.9
CategoricalNB 의 정답률 :  0.9333333333333333
CheckingClassifier 의 정답률 :  0.26666666666666666
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.7333333333333333
DecisionTreeClassifier 의 정답률 :  0.8666666666666667
DummyClassifier 의 정답률 :  0.2
ExtraTreeClassifier 의 정답률 :  0.7666666666666667
ExtraTreesClassifier 의 정답률 :  0.9
GaussianNB 의 정답률 :  0.9333333333333333
GaussianProcessClassifier 의 정답률 :  0.9
GradientBoostingClassifier 의 정답률 :  0.9
HistGradientBoostingClassifier 의 정답률 :  0.9
KNeighborsClassifier 의 정답률 :  0.9333333333333333
LabelPropagation 의 정답률 :  0.9666666666666667
LabelSpreading 의 정답률 :  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9333333333333333
LogisticRegression 의 정답률 :  0.9333333333333333
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.6666666666666666
NearestCentroid 의 정답률 :  0.9
NuSVC 의 정답률 :  0.9
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  1.0
Perceptron 의 정답률 :  0.4666666666666667
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
RandomForestClassifier 의 정답률 :  0.8666666666666667
RidgeClassifier 의 정답률 :  0.8666666666666667
RidgeClassifierCV 의 정답률 :  0.8666666666666667
SGDClassifier 의 정답률 :  0.4666666666666667
SVC 의 정답률 :  0.9
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''