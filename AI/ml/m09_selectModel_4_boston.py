# type_filter='regressor'
import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_boston

datasets=load_boston()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=33)

allAlgorithms=all_estimators(type_filter='regressor') # 회귀형 모델 전체
# all_estimators - sklearn 0.20.0 대에 최적화
print(sklearn.__version__) # 0.23.2

for (name, algorithm) in allAlgorithms: # name : 분류모델의 이름, algorithm : 분류모델
    try:
        model=algorithm()

        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except: # 예외 발생시 실행 할 행동
        print(name, '은 없는 모델') # continue 를 쓰면 무시하고 루프를 계속 돌게 됨


'''
0.23.2 에서 쓸 수 있는 모델들

tensorflow r2 : 0.93280876980413

ARDRegression 의 정답률 :  0.6780654234186746
AdaBoostRegressor 의 정답률 :  0.776194166401695
BaggingRegressor 의 정답률 :  0.7873230948251595
BayesianRidge 의 정답률 :  0.6739193815224682
CCA 의 정답률 :  0.5576945849111905
DecisionTreeRegressor 의 정답률 :  0.6534503223136072
DummyRegressor 의 정답률 :  -0.031660985575741485
ElasticNet 의 정답률 :  0.6572152690673503
ElasticNetCV 의 정답률 :  0.6511781759646544
ExtraTreeRegressor 의 정답률 :  0.6962575163023874
ExtraTreesRegressor 의 정답률 :  0.8561258157520168
GammaRegressor 의 정답률 :  -0.03166098557574171
GaussianProcessRegressor 의 정답률 :  -6.268062458366597
GeneralizedLinearRegressor 의 정답률 :  0.6357080657222156
GradientBoostingRegressor 의 정답률 :  0.8460120732954517
HistGradientBoostingRegressor 의 정답률 :  0.8266449770654344
HuberRegressor 의 정답률 :  0.6608858502164057
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 정답률 :  0.6287611999602677
KernelRidge 의 정답률 :  0.6380510278766541
Lars 의 정답률 :  0.6904233544589051
LarsCV 의 정답률 :  0.694605245686085
Lasso 의 정답률 :  0.6592074324274
LassoCV 의 정답률 :  0.6718127236890896
LassoLars 의 정답률 :  -0.031660985575741485
LassoLarsCV 의 정답률 :  0.6947879909064452
LassoLarsIC 의 정답률 :  0.6941895120920321
LinearRegression 의 정답률 :  0.6922908805512095
LinearSVR 의 정답률 :  0.41629853589365995
MLPRegressor 의 정답률 :  0.332126247305081
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 정답률 :  0.3425415759545364
OrthogonalMatchingPursuit 의 정답률 :  0.529297335443208
OrthogonalMatchingPursuitCV 의 정답률 :  0.6172155412505074
PLSCanonical 의 정답률 :  -3.777796560776414
PLSRegression 의 정답률 :  0.6682815688704427
PassiveAggressiveRegressor 의 정답률 :  -1.1679063112342427
PoissonRegressor 의 정답률 :  0.7291992765601093
RANSACRegressor 의 정답률 :  -0.27578665395079094
RadiusNeighborsRegressor 은 없는 모델
RandomForestRegressor 의 정답률 :  0.8356724285652815
RegressorChain 은 없는 모델
Ridge 의 정답률 :  0.6833396417695796
RidgeCV 의 정답률 :  0.6910916738785535
SGDRegressor 의 정답률 :  -2.8716544035278645e+26
SVR 의 정답률 :  0.31226916707634644
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 :  0.6736380992958029
TransformedTargetRegressor 의 정답률 :  0.6922908805512095
TweedieRegressor 의 정답률 :  0.6357080657222156
VotingRegressor 은 없는 모델
_SigmoidCalibration 은 없는 모델
'''