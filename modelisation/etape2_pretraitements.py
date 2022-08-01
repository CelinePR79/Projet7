#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: celine
"""

result_path_base = 'Resultats/'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#====================================================================
# Découverte du dataframe « light »
#====================================================================

df_light=pd.read_csv(result_path_base + "data_light.csv", sep=',')

df_light.shape
#(356251, 798)


#-- Les features utilisées
features_importances=pd.read_csv(result_path_base + "df_features_importances.csv", sep=',')
features_importances_top10 = features_importances[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:10]

# sans la colonne 'TARGET'
list_features=features_importances_top10.index.to_list()

# avec la colonne 'TARGET'
list_features_target=['TARGET']+list_features

#['TARGET','PAYMENT_RATE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_ID_PUBLISH','ACTIVE_DAYS_CREDIT_MAX','ACTIVE_DAYS_CREDIT_ENDDATE_MIN']


# Description des features utilisées   
    
for ele in list_features:
    print(ele)
    print(data6[data6['Row']==ele]['Description'])
	Description
    
#EXT_SOURCE_1    Normalized score from external data source
#EXT_SOURCE_2    Normalized score from external data source
#EXT_SOURCE_3    Normalized score from external data source
#DAYS_BIRTH      Client's age in days at the time of application
# AMT_ANNUITY
# 9                             Loan annuity
# 138    Annuity of the Credit Bureau credit
# 176        Annuity of previous application
# # DAYS_EMPLOYED	How many days before the application the person started current employment
# DAYS_ID_PUBLISH    	How many days before the application did client change the identity document with which he applied for the loan
# ACTIVE_DAYS_CREDIT_MAX  How many days before current application did client apply for Credit Bureau credit
# ACTIVE_DAYS_CREDIT_ENDDATE_MIN Remaining duration of CB credit (in days) at the time of application in Home Credit


##-- les data avec seulement les 10 features les plus importantes
df_10_features=df_light[list_features_target]

df_10_features.info()
# TODO info

plt.figure(figsize=(20,10))
sns.heatmap(df_10_features.isna(),cbar=False)
plt.show()

##-- Calcul du % de valeurs non NaN
def ratioValeursExploitables(dataframe) :
    nbCellules = dataframe.shape[0] * dataframe.shape[1] # nb lignes * nb colonnes
    # dataframe.count() => nb non NaN / colonne --> type Series
    nbCellulesNonNaN = dataframe.count().sum()
    ratio = nbCellulesNonNaN / nbCellules
    return ratio

ratio = ratioValeursExploitables(df_10_features)
print("Ratio valeurs utiles dans le df : ", ratio)





#====================================================================
# Nettoyages des valeurs NAN
#====================================================================

##-- Traitement de la colonne TARGET

df_10_features['TARGET'].value_counts(dropna=False)
# 0.0    282682
# NaN     48744
# 1.0     24825

# Suppression des NaN de la colonne TARGET
df_10_features=df_10_features.dropna(subset=['TARGET'])

# exploration
df_10_features['TARGET'].describe()
df_10_features['TARGET'].shape


##-- Traitement des features

# Description de chaque feature et remplacement des valeurs manquantes par la moyenne
def description_features(data,feature_name):
    describe_feature=data[feature_name].describe()
    repartition_feature=data[feature_name].value_counts(dropna=False)
    data[feature_name]=data[feature_name].fillna(data[feature_name].mean())
    print(describe_feature)
    print(repartition_feature)
    print('fillna Done.')

for ele in list_features:
    description_features(df_10_features,ele)


##-- Suppression des lignes vides

## REMARQUE : en effectuant les traitements avec le DataFrame nettoyé et imputé, les temps de traitement étaient beaucoup trop long ou n'aboutissaient pas. J'ai donc décidé de supprimer les lignes vides

df_10_features=df_10_features.dropna(subset=list_features_target)

df_10_features.shape
#(81468, 11)
# taille réduit à environ 80000lignes



#====================================================================
# Nettoyages des outliers
# 2 methodes :
#   - Interquartiles
#   - Moy+/-3*Ecart type
#====================================================================

#--------------------------------------------------------------------
# Méthode interquartile
#--------------------------------------------------------------------

Q1=df_10_features.quantile(0.25)
Q3=df_10_features.quantile(0.75)
print('Q1 25 percentile of the given data is, ', Q1)
print('Q1 75 percentile of the given data is, ', Q3)

IQR = Q3 - Q1 
print('Interquartile range is', IQR)

low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR
print('low_limit is', low_lim)
print('up_limit is', up_lim)


# combien d'éléments en dehors des limites (inf et sup)
out_col =[]
for col in list_features:
    for ele in df_10_features[col]:
        if ((ele> up_lim[col]) or (ele<low_lim[col])):
            out_col.append(ele)

chgt1=len(out_col)/(df_10_features[list_features].shape[0]*df_10_features[list_features].shape[1])*100

print('Le pourcentage de changement serait de : ', chgt1)
#Le pourcentage de changement serait de :  3.672362154465557



#--------------------------------------------------------------------
# Méthode Moy+/-3*Ecart type
#--------------------------------------------------------------------

M1=df_10_features.mean()
print('Q1 25 percentile of the given data is, ', M1)
STD=df_10_features.std()
print('Q1 25 percentile of the given data is, ', STD)

lim_inf=M1-3*STD
lim_sup=M1+3*STD

# combien d'éléments en dehors des limites (inf et sup)
out_col2 =[]
for col in list_features:
    for ele in df_10_features[col]:
        if ((ele> lim_sup[col]) or (ele<lim_inf[col])):
            out_col2.append(ele)
            
            
chgt2=len(out_col2)/(df_10_features[list_features].shape[0]*df_10_features[list_features].shape[1])*100

print('Le pourcentage de changement serait de : ', chgt2)
#Le pourcentage de changement serait de :  0.8636519860558747

## ==> methode retenue 



#--------------------------------------------------------------------
# Application de la méthode Moy+/-3*Ecart type
#--------------------------------------------------------------------

#Remplace les outliers methode moy+/-3* std par la median
for ele in list_features:
    df_10_features.loc[(df_10_features[ele]>lim_sup[ele])|(df_10_features[ele]<lim_inf[ele]),ele]=df_10_features[ele].median()

out_col_check =[]
for col in list_features:
    for ele in df_10_features[col]:
        if ((ele> lim_sup[col]) or (ele<lim_inf[col])):
            out_col_check.append(ele)

# on vérifie que la liste est vide (donc pas d'éléméent en dehors des limites)
len(out_col_check)





#====================================================================
# Analyse des données nettoyées
# et sauvegarde
#====================================================================

##-- Analyse univariée

for ele in list_features:
    sns.displot(x=ele,data=df_10_features,kind='kde',fill=True,lw=3)

##--  Analyse bivarie

sns.pairplot(df_10_features[1::])
plt.show()

sns.heatmap(df_10_features.iloc[:,1:11].corr(),annot=True)
plt.show()


##--  sauvegarde

df_10_features.to_csv(result_path_base + "df_10_features_80.csv",index=False)

