import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators

from sklearn.datasets import load_diabetes

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=11)

all_algorithms=all_estimators(type_filter='regressor')

for (name, algoritm) in all_algorithms:
    try:
        model=algoritm()

        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(name, '의 정답률', r2_score(y_test, y_pred))

    except:
        print(name, '은 없는 모델')


'''
tensorflow r2 : 0.5114865328386683

ARDRegression 의 정답률 0.5781341488645728
AdaBoostRegressor 의 정답률 0.5232823483313207
BaggingRegressor 의 정답률 0.5114593539746259
BayesianRidge 의 정답률 0.5706905540229448
CCA 의 정답률 0.5377515324927817
DecisionTreeRegressor 의 정답률 0.11230317357872166
DummyRegressor 의 정답률 -0.0007989085919366534
ElasticNet 의 정답률 0.007350189238993665
ElasticNetCV 의 정답률 0.5075682327612228
ExtraTreeRegressor 의 정답률 0.04476459224534646
ExtraTreesRegressor 의 정답률 0.5711955261328978
GammaRegressor 의 정답률 0.005556153553387899
GaussianProcessRegressor 의 정답률 -5.455298214272385
GeneralizedLinearRegressor 의 정답률 0.005623719153655005
GradientBoostingRegressor 의 정답률 0.5383825716688371
HistGradientBoostingRegressor 의 정답률 0.5420121322703939
HuberRegressor 의 정답률 0.5632350004035762
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 정답률 0.47789414310263956
KernelRidge 의 정답률 -2.5896215026292553
Lars 의 정답률 0.20133078084881528
LarsCV 의 정답률 0.5718053159911549
Lasso 의 정답률 0.35466242767381206
LassoCV 의 정답률 0.5721003962539936
LassoLars 의 정답률 0.38954423078245015
LassoLarsCV 의 정답률 0.572326469713454
LassoLarsIC 의 정답률 0.5718044956267778
LinearRegression 의 정답률 0.5771693213852914
LinearSVR 의 정답률 -0.2690605230206271
MLPRegressor 의 정답률 -2.5212427607286183
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 정답률 0.17561041589916404
OrthogonalMatchingPursuit 의 정답률 0.4118345919808919
OrthogonalMatchingPursuitCV 의 정답률 0.5837755288546551
PLSCanonical 의 정답률 -0.25381218248653803
PLSRegression 의 정답률 0.5760863799566744
PassiveAggressiveRegressor 의 정답률 0.5431698539464557
PoissonRegressor 의 정답률 0.378263171137991
RANSACRegressor 의 정답률 0.22879441444334492
RadiusNeighborsRegressor 의 정답률 -0.0007989085919366534
RandomForestRegressor 의 정답률 0.5734196205657799
RegressorChain 은 없는 모델
Ridge 의 정답률 0.47588972376223304
RidgeCV 의 정답률 0.5673623564173929
SGDRegressor 의 정답률 0.47455120680439655
SVR 의 정답률 0.1776275183192425
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 0.565705383707706
TransformedTargetRegressor 의 정답률 0.5771693213852914
TweedieRegressor 의 정답률 0.005623719153655005
VotingRegressor 은 없는 모델
_SigmoidCalibration 은 없는 모델
'''