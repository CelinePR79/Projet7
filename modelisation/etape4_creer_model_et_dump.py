#!/usr/bin/env python
# coding: utf-8

result_path_base = 'Resultats/'

## OBJECTIF : créer le modèle Random Forest

from sklearn import preprocessing, model_selection, ensemble, pipeline
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# chargement des data nettoyées

df_80 = pd.read_csv(result_path_base + 'df_10_features_80.csv', sep=',')
list_features_only=['PAYMENT_RATE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_ID_PUBLISH','ACTIVE_DAYS_CREDIT_MAX','ACTIVE_DAYS_CREDIT_ENDDATE_MIN']

X = df_80[list_features_only]  # V1 et V3
#X = np.array( df_80[list_features_only] ) # V2, mais perte des en-têtes lors de l'enregistrement en CSV
y = df_80['TARGET']
#X.shape, y.shape
# ((81468, 10), (81468,))

# un exemple de client
#client = df_80[list_features_only].sample()
#client



### Modèle

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42) # train_size=0.8 ou bien random_state=42 pour avoir toujours le même découpage (pratique pour deboguer !!)

X_train.to_csv(result_path_base + 'X_train_avec-index.csv', index=True)
y_train.to_csv(result_path_base + 'y_train_avec-index.csv', index=True)
X_test.to_csv (result_path_base + 'X_test_avec-index.csv',  index=True)
y_test.to_csv (result_path_base + 'y_test_avec-index.csv',  index=True)

X_train.to_csv(result_path_base + 'X_train.csv', index=False)
y_train.to_csv(result_path_base + 'y_train.csv', index=False)
X_test.to_csv (result_path_base + 'X_test.csv',  index=False)
y_test.to_csv (result_path_base + 'y_test.csv',  index=False)



## Trouver les best_params_ pour RandomForest

import imblearn
#print(imblearn.__version__)
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import fbeta_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
ftwo_scorer = make_scorer(fbeta_score, beta=2)

param_grid_rf = {
    'bootstrap': [True],
    'max_depth': [30,80],
    'max_features': [ 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10],
    'n_estimators': [300, 1000]
}

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# rééquilibre le dataset
from imblearn.over_sampling import SMOTE 
oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X_train_scaled, y_train)

rf = RandomForestClassifier()
GS_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, cv = 3, n_jobs = -1, verbose = 2, scoring=ftwo_scorer)
GS_rf.fit(X_res, y_res)
GS_rf.best_score_, GS_rf.best_params_

GS_rf.best_estimator_.score(X_test_scaled, y_test)


## Faire le modèle avec pipeline et best_params_

from imblearn.pipeline import Pipeline as imbpipeline

pipe = imbpipeline([('scaler', preprocessing.StandardScaler()),
                   ('balance', SMOTE()),
                   ('clf', ensemble.RandomForestClassifier(**GS_rf.best_params_))])

pipe.fit(X_train, y_train)
train_accuracy = pipe.score(X_train, y_train)
test_accuracy  = pipe.score(X_test, y_test)


## Test sur un individu
#pipeline.predict(df_80[list_features_only].sample())


## precision    recall  f1-score

#y_pred=pipe.predict(X_test)
#from sklearn.metrics import classification_report
#target_names = ['0', '1']
#print(classification_report(y_test, y_pred, target_names=target_names))

              #precision    recall  f1-score   support
#weighted avg       0.88      0.85      0.86     12500



### Sauvegarde du modèle

import joblib
#from joblib import dump, load

## Enregistrement
# compress - from 0 to 9. Higher value means more compression, but also slower read and write times. Using a value of 3 is often a good compromise.
joblib.dump(pipe, result_path_base + 'pipeline_credit_compress3.joblib', compress=3)
joblib.dump(pipe, result_path_base + 'pipeline_credit_compress9.joblib', compress=9)

## Chargement
#clf = joblib.load(result_path_base + 'pipeline_credit.joblib')

# test sur un individu
#clf.predict(df_80[list_features_only].sample())

