import pandas as pd
import numpy as np

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

import sklearn.metrics as metrics

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

random_state = 23

df = pd.read_csv('cmpd.csv')

df['mol'] = df.smiles.apply(Chem.MolFromSmiles)
df['mol'] = df.mol.apply(Chem.AddHs)

# print(type(df['smiles'][0]))

print(df.describe())
for idx in range(len(df)):
    temp_list = []
    temp_list.append(df.iloc[idx, 0].split("-"))
    df['num_of_atoms'] = df['mol'][idx].GetNumAtoms()
    df['num_of_heavy_atoms'] = df['mol'][idx].GetNumHeavyAtoms()
    df['inchikey_1'] = temp_list[0][0]
    df['inchikey_2'] = temp_list[0][1]

df = df[['inchikey', 'inchikey_1', 'inchikey_2', 'smiles', 'group', 'activity', 'mol', 'num_of_atoms', 'num_of_heavy_atoms']]

def get_Xy(df):
    X = np.vstack(df.mol.apply(lambda m: list(AllChem.GetMorganFingerprintAsBitVect(m, 9, nBits=2048))))
    y = df.activity.eq('active').astype(float).to_numpy()
    return X, y

print(df.describe())
print(df.head())


X_train, y_train = get_Xy(df[df.group.eq('train')])
X_test, y_test = get_Xy(df[df.group.eq('test')])

kf = KFold(n_splits=10, shuffle=True)
skf = StratifiedKFold(n_splits=10, shuffle=True)

model = XGBClassifier(
    n_estimator=1000,
    n_jobs=-1,
    max_depth=10
)

# score = cross_val_score(model, X_train, y_train, cv=kf)
# score_2 = cross_val_score(model, X_train, y_train, cv=skf)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

y_pred = model.predict_proba(X_test)[:, 1]

log_loss = metrics.log_loss(y_test, y_pred, labels=[0, 1])

precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred, pos_label=1)
auc1 = metrics.auc(recall, precision)

fpr_roc, tpr_roc, _ = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc2 = metrics.auc(fpr_roc, tpr_roc)

print(f"score : {score}")
print(f"log_loss : {log_loss}")
print(f"auc1  : {auc1}")
print(f"auc2 : {auc2}")

# with inchikey_1, 2
# score : 0.8164842240824212
# log_loss : 0.4389295898857633
# auc1  : 0.8521998563755077
# auc2 : 0.8729480737018425

# without inchikey_1, 2
# score : 0.8164842240824212
# log_loss : 0.4389295898857633
# auc1  : 0.8521998563755077
# auc2 : 0.8729480737018425

# with num_of_atoms, num_of_heavy_atoms
# score : 0.8254990341274951
# log_loss : 0.41599603853977446
# auc1  : 0.8586843880390641
# auc2 : 0.8799256137317412