import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as metrics

cmpd_df = pd.read_csv('cmpd.csv')

print(cmpd_df.head())
print(cmpd_df.shape) # (5530, 4)

cmpd_df['mol'] = cmpd_df.smiles.apply(Chem.MolFromSmiles)

print(cmpd_df.head())

def get_Xy(df):
    X = np.vstack(df.mol.apply(lambda m: list(AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=2048))))
    y = df.activity.eq('active').astype(float).to_numpy()
    return X, y

X_train, y_train = get_Xy(cmpd_df[cmpd_df.group.eq('train')])
X_test, y_test = get_Xy(cmpd_df[cmpd_df.group.eq('test')])

print(X_train)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

y_pred = clf.predict_proba(X_test)[:, 1]

print(metrics.log_loss(y_test, y_pred, labels=[0, 1]))

precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred, pos_label=1)
print(metrics.auc(recall, precision))

fpr_roc, tpr_roc, _ = metrics.roc_curve(y_test, y_pred, pos_label=1)
print(metrics.auc(fpr_roc, tpr_roc))