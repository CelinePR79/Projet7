#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: celine
"""

result_path_base = 'Resultats/'

## OBJECTIF : Test des algo GB, RF, Regressin logistic et Isolation Forest, Dummy classifier

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets, preprocessing, model_selection, ensemble, pipeline

#====================================================================
# Split et Standardisation des données
#====================================================================

##-- chargement
df_10_features=pd.read_csv(result_path_base + "df_10_features_80.csv", sep=',')

# features (sans la colonne 'TARGET')
list_features=['PAYMENT_RATE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_ID_PUBLISH','ACTIVE_DAYS_CREDIT_MAX','ACTIVE_DAYS_CREDIT_ENDDATE_MIN']

X=np.array(df_10_features[list_features])
y=df_10_features['TARGET']

##-- Découpage du jeu de données Train / Test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8)

X_train.shape, X_test.shape


##-- Standardisation des données
scaler = preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#====================================================================
# Rééquilibrage des données, avec SMOTE
#====================================================================

import collections
import imblearn
#print(imblearn.__version__)


# summarize class distribution
counter = collections.Counter(y)
print(counter)

# scatter plot of examples by class label before rééquilibrage
for label, _ in counter.items():
	row_ix = np.where(y_train == label)[0]
	plt.scatter(X_train_scaled[row_ix, 0], X_train_scaled[row_ix, 1], label=str(label))
plt.legend()
plt.title('class before oversampling')
plt.show()


##-- transform the dataset

from imblearn.over_sampling import SMOTE 
oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X_train_scaled, y_train)
X_res.shape, X_test.shape

# summarize the NEW class distribution
counter = collections.Counter(y_res)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = np.where(y_res == label)[0]
	plt.scatter(X_res[row_ix, 0], X_res[row_ix, 1], label=str(label))
plt.title('class after oversampling')    
plt.legend()
plt.show()



#====================================================================
# Modélisation selon différentes approches
#====================================================================

#--------------------------------------------------------------------
# Gradient Boosting
#--------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

##-- fit du modèle

classifier_gb = GradientBoostingClassifier()
params_gb = {'learning_rate': [1e-1, 2e-1, 3e-1],
          'max_depth': [2, 4, 5],
          'n_estimators': [100, 200, 300, 500]
          }
gs_gb = GridSearchCV(estimator=classifier_gb, param_grid=params_gb, cv=5, scoring='roc_auc')
gs_gb.fit(X_res, y_res)


##--  Affichage des meilleurs scores et paramètres

gs_gb.best_score_, gs_gb.best_params_
# (0.8931761018608524,
#  {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 500})


##-- courbe ROC

from sklearn.metrics import roc_curve, auc

y_prob_gb=gs_gb.predict_proba(X_test_scaled)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_gb)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)


y_prob_gb2=gs_gb.predict_proba(X_res)[:,1]
false_positive_rate2, true_positive_rate2, thresholds2 = roc_curve(y_res, y_prob_gb2)
roc_auc2 = auc(false_positive_rate2, true_positive_rate2)
print(roc_auc2)


# Afficher les courbes ROC

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC_test = %0.2f' % roc_auc)
plt.plot(false_positive_rate2,true_positive_rate2, color='green',label = 'AUC_train = %0.2f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Gradient Boosting')
plt.show()

#plt.savefig(result_path_base+'GradientBoosting.png', bbox_inches="tight")


#y_prob_proba = gs_gb.predict_proba(X_test_scaled)[:, 1]
#[fpr, tpr, thr] = roc_curve(y_test, y_prob_proba)
#plt.plot(fpr, tpr, color='coral', lw=2)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('1 - specificite', fontsize=14)
#plt.ylabel('Sensibilite', fontsize=14)


#print(auc(fpr, tpr))
#idx = np.min(np.where(tpr > 0.95)) # indice du premier seuil pour lequel la sensibilité est supérieure à 0.95

#print("Sensibilité : {:.2f}".format(tpr[idx]))
#print("Spécificité : {:.2f}".format(1-fpr[idx]))
#print("Seuil : {:.2f}".format(thr[idx]))



#--------------------------------------------------------------------
# Randon Forest
#--------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

##-- fit du modèle

classifier_rf = RandomForestClassifier()
param_rf = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
gs_rf = GridSearchCV(estimator = classifier_rf, 
                          param_grid = param_rf, 
                          cv = 5, n_jobs = -1, verbose = 2, scoring='roc_auc')
gs_rf.fit(X_res, y_res)


##--  Affichage des meilleurs scores et paramètres

gs_rf.best_score_, gs_rf.best_params_
# (0.9028627046809892,
#  {'bootstrap': True,
#   'max_depth': 80,
#   'max_features': 3,
#   'min_samples_leaf': 3,
#   'min_samples_split': 8,
#   'n_estimators': 1000})


##-- courbe ROC

from sklearn.metrics import roc_curve, auc

y_prob_rf=gs_rf.predict_proba(X_test_scaled)[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

# Afficher la courbe ROC

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Randon Forest')
plt.show()

#plt.savefig(result_path_base+'RandonForest.png', bbox_inches="tight")



#--------------------------------------------------------------------
# Regression Logistic
#--------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

##-- fit du modèle

classifier_lr = LogisticRegression(solver = 'liblinear')
param_lr = {'C': np.logspace(-3, 3, 7) , 'penalty':['l1','l2'] }

gs_lr = GridSearchCV(estimator = classifier_lr, param_grid = param_lr, cv=10,scoring='roc_auc')
gs_lr.fit(X_res, y_res)


##--  Affichage des meilleurs scores et paramètres

print(gs_lr.best_params_)
#Output from spyder call 'get_namespace_view':
#{'C': 0.01, 'penalty': 'l2'}


##-- courbe ROC

from sklearn.metrics import roc_curve, auc

y_prob_lr = gs_lr.predict_proba(X_test_scaled)[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_lr)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)


# Afficher la courbe ROC

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Regression logistique')
plt.show()

#plt.savefig(result_path_base+'RegressionLogistique.png', bbox_inches="tight")



#--------------------------------------------------------------------
# Isolation Forest
#--------------------------------------------------------------------

from sklearn.ensemble import IsolationForest

##-- fit du modèle

classifier_if = IsolationForest(contamination=0.01)
classifier_if.fit(X_res)

##--  Affichage des meilleurs scores et paramètres
classifier_if.best_estimator_.score(X_test_scaled, y_test)


##-- courbe ROC

from sklearn.metrics import roc_curve, auc

y_prob_if = classifier_if.fit(X_res).decision_function(X_test_scaled)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_if)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

# Afficher la courbe ROC

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC_test = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Isolation Forest')
plt.show()

#plt.savefig(result_path_base+'IsolationForest.png', bbox_inches="tight")



#--------------------------------------------------------------------
# Dummy Classifier
#--------------------------------------------------------------------

from sklearn import dummy

##-- fit du modèle

#Resultat avec GridSearchCV sur la strategy : la meilleure est 'stratified'

#classifier_dummy = dummy.DummyClassifier()
#param_dummy = {'strategy':['most_frequent','stratified','uniform'] }

#gs_dummy = GridSearchCV(estimator = classifier_dummy, param_grid = param_dummy, cv=5,scoring='roc_auc')
#gs_dummy.fit(X_res, y_res)

#print(gs_dummy.best_params_)
#{'strategy': 'stratified'}


classifier_dum = dummy.DummyClassifier(strategy='stratified')

classifier_dum.fit(X_res, y_res)

##--  Affichage des meilleurs scores et paramètres
classifier_dum.best_estimator_.score(X_test_scaled, y_test)



##-- courbe ROC

from sklearn.metrics import roc_curve, auc

y_prob_dum = dum.predict_proba(X_test_scaled)[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_dum)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)


# Afficher la courbe ROC

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Dummy classifier')
plt.show()

#plt.savefig(result_path_base+'DummyClassifier.png', bbox_inches="tight")



#--------------------------------------------------------------------
# Choix du modèle
#--------------------------------------------------------------------

# meilleurs ROC : 0.73 pour RF et 0.75 pour RL
