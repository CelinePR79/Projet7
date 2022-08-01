#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: celine
"""

data_path_base = 'Data_brut/'
result_path_base = 'Resultats/'

#====================================================================
# Découverte des données
#====================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
#get_ipython().run_line_magic('matplotlib', 'inline')
from os import listdir

data1=pd.read_csv(data_path_base + "application_test.csv", sep=',')  
data2=pd.read_csv(data_path_base + "application_train.csv", sep=',')  
data3=pd.read_csv(data_path_base + "bureau_balance.csv", sep=',')  
data4=pd.read_csv(data_path_base + "bureau.csv", sep=',')  
data5=pd.read_csv(data_path_base + "credit_card_balance.csv", sep=',')  
data6=pd.read_csv(data_path_base + "HomeCredit_columns_description.csv", encoding = 'ISO-8859-1', sep=',')  
data7=pd.read_csv(data_path_base + "installments_payments.csv", sep=',')  
data8=pd.read_csv(data_path_base + "POS_CASH_balance.csv", sep=',')  
data9=pd.read_csv(data_path_base + "previous_application.csv", sep=',')  
data10=pd.read_csv(data_path_base + "sample_submission.csv", sep=',')  

data_brut_all=[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]

# Trop long
# for ele in data_brut_all:
#     plt.figure(figsize=(20,10))
#     sns.heatmap(ele.isna(),cbar=False)
#     plt.show()


##-- Calcul du % de valeurs non NaN
def ratioValeursExploitables(dataframe) :
    nbCellules = dataframe.shape[0] * dataframe.shape[1] # nb lignes * nb colonnes
    # dataframe.count() => nb non NaN / colonne --> type Series
    nbCellulesNonNaN = dataframe.count().sum()
    ratio = nbCellulesNonNaN / nbCellules
    return ratio

for ele in data_brut_all:
    ratio = ratioValeursExploitables(ele)
    print("Ratio valeurs utiles dans le df : ", ratio)

# Ratio valeurs utiles dans le df :  0.7618831323846766
# Ratio valeurs utiles dans le df :  0.7560405809287056
# Ratio valeurs utiles dans le df :  1.0
# Ratio valeurs utiles dans le df :  0.8649744770912068
# Ratio valeurs utiles dans le df :  0.9334592560731252
# Ratio valeurs utiles dans le df :  0.8785388127853881
# Ratio valeurs utiles dans le df :  0.9999466204634468
# Ratio valeurs utiles dans le df :  0.9993481135261831
# Ratio valeurs utiles dans le df :  0.8202312252655503
# Ratio valeurs utiles dans le df :  1.0

# Information
for ele in data_brut_all:
    ele.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 48744 entries, 0 to 48743
# Columns: 121 entries, SK_ID_CURR to AMT_REQ_CREDIT_BUREAU_YEAR
# dtypes: float64(65), int64(40), object(16)
# memory usage: 45.0+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 307511 entries, 0 to 307510
# Columns: 122 entries, SK_ID_CURR to AMT_REQ_CREDIT_BUREAU_YEAR
# dtypes: float64(65), int64(41), object(16)
# memory usage: 286.2+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 27299925 entries, 0 to 27299924
# Data columns (total 3 columns):
#  #   Column          Dtype 
# ---  ------          ----- 
#  0   SK_ID_BUREAU    int64 
#  1   MONTHS_BALANCE  int64 
#  2   STATUS          object
# dtypes: int64(2), object(1)
# memory usage: 624.8+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1716428 entries, 0 to 1716427
# Data columns (total 17 columns):
#  #   Column                  Dtype  
# ---  ------                  -----  
#  0   SK_ID_CURR              int64  
#  1   SK_ID_BUREAU            int64  
#  2   CREDIT_ACTIVE           object 
#  3   CREDIT_CURRENCY         object 
#  4   DAYS_CREDIT             int64  
#  5   CREDIT_DAY_OVERDUE      int64  
#  6   DAYS_CREDIT_ENDDATE     float64
#  7   DAYS_ENDDATE_FACT       float64
#  8   AMT_CREDIT_MAX_OVERDUE  float64
#  9   CNT_CREDIT_PROLONG      int64  
#  10  AMT_CREDIT_SUM          float64
#  11  AMT_CREDIT_SUM_DEBT     float64
#  12  AMT_CREDIT_SUM_LIMIT    float64
#  13  AMT_CREDIT_SUM_OVERDUE  float64
#  14  CREDIT_TYPE             object 
#  15  DAYS_CREDIT_UPDATE      int64  
#  16  AMT_ANNUITY             float64
# dtypes: float64(8), int64(6), object(3)
# memory usage: 222.6+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3840312 entries, 0 to 3840311
# Data columns (total 23 columns):
#  #   Column                      Dtype  
# ---  ------                      -----  
#  0   SK_ID_PREV                  int64  
#  1   SK_ID_CURR                  int64  
#  2   MONTHS_BALANCE              int64  
#  3   AMT_BALANCE                 float64
#  4   AMT_CREDIT_LIMIT_ACTUAL     int64  
#  5   AMT_DRAWINGS_ATM_CURRENT    float64
#  6   AMT_DRAWINGS_CURRENT        float64
#  7   AMT_DRAWINGS_OTHER_CURRENT  float64
#  8   AMT_DRAWINGS_POS_CURRENT    float64
#  9   AMT_INST_MIN_REGULARITY     float64
#  10  AMT_PAYMENT_CURRENT         float64
#  11  AMT_PAYMENT_TOTAL_CURRENT   float64
#  12  AMT_RECEIVABLE_PRINCIPAL    float64
#  13  AMT_RECIVABLE               float64
#  14  AMT_TOTAL_RECEIVABLE        float64
#  15  CNT_DRAWINGS_ATM_CURRENT    float64
#  16  CNT_DRAWINGS_CURRENT        int64  
#  17  CNT_DRAWINGS_OTHER_CURRENT  float64
#  18  CNT_DRAWINGS_POS_CURRENT    float64
#  19  CNT_INSTALMENT_MATURE_CUM   float64
#  20  NAME_CONTRACT_STATUS        object 
#  21  SK_DPD                      int64  
#  22  SK_DPD_DEF                  int64  
# dtypes: float64(15), int64(7), object(1)
# memory usage: 673.9+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 219 entries, 0 to 218
# Data columns (total 5 columns):
#  #   Column       Non-Null Count  Dtype 
# ---  ------       --------------  ----- 
#  0   Unnamed: 0   219 non-null    int64 
#  1   Table        219 non-null    object
#  2   Row          219 non-null    object
#  3   Description  219 non-null    object
#  4   Special      86 non-null     object
# dtypes: int64(1), object(4)
# memory usage: 8.7+ KB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 13605401 entries, 0 to 13605400
# Data columns (total 8 columns):
#  #   Column                  Dtype  
# ---  ------                  -----  
#  0   SK_ID_PREV              int64  
#  1   SK_ID_CURR              int64  
#  2   NUM_INSTALMENT_VERSION  float64
#  3   NUM_INSTALMENT_NUMBER   int64  
#  4   DAYS_INSTALMENT         float64
#  5   DAYS_ENTRY_PAYMENT      float64
#  6   AMT_INSTALMENT          float64
#  7   AMT_PAYMENT             float64
# dtypes: float64(5), int64(3)
# memory usage: 830.4 MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10001358 entries, 0 to 10001357
# Data columns (total 8 columns):
#  #   Column                 Dtype  
# ---  ------                 -----  
#  0   SK_ID_PREV             int64  
#  1   SK_ID_CURR             int64  
#  2   MONTHS_BALANCE         int64  
#  3   CNT_INSTALMENT         float64
#  4   CNT_INSTALMENT_FUTURE  float64
#  5   NAME_CONTRACT_STATUS   object 
#  6   SK_DPD                 int64  
#  7   SK_DPD_DEF             int64  
# dtypes: float64(2), int64(5), object(1)
# memory usage: 610.4+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1670214 entries, 0 to 1670213
# Data columns (total 37 columns):
#  #   Column                       Non-Null Count    Dtype  
# ---  ------                       --------------    -----  
#  0   SK_ID_PREV                   1670214 non-null  int64  
#  1   SK_ID_CURR                   1670214 non-null  int64  
#  2   NAME_CONTRACT_TYPE           1670214 non-null  object 
#  3   AMT_ANNUITY                  1297979 non-null  float64
#  4   AMT_APPLICATION              1670214 non-null  float64
#  5   AMT_CREDIT                   1670213 non-null  float64
#  6   AMT_DOWN_PAYMENT             774370 non-null   float64
#  7   AMT_GOODS_PRICE              1284699 non-null  float64
#  8   WEEKDAY_APPR_PROCESS_START   1670214 non-null  object 
#  9   HOUR_APPR_PROCESS_START      1670214 non-null  int64  
#  10  FLAG_LAST_APPL_PER_CONTRACT  1670214 non-null  object 
#  11  NFLAG_LAST_APPL_IN_DAY       1670214 non-null  int64  
#  12  RATE_DOWN_PAYMENT            774370 non-null   float64
#  13  RATE_INTEREST_PRIMARY        5951 non-null     float64
#  14  RATE_INTEREST_PRIVILEGED     5951 non-null     float64
#  15  NAME_CASH_LOAN_PURPOSE       1670214 non-null  object 
#  16  NAME_CONTRACT_STATUS         1670214 non-null  object 
#  17  DAYS_DECISION                1670214 non-null  int64  
#  18  NAME_PAYMENT_TYPE            1670214 non-null  object 
#  19  CODE_REJECT_REASON           1670214 non-null  object 
#  20  NAME_TYPE_SUITE              849809 non-null   object 
#  21  NAME_CLIENT_TYPE             1670214 non-null  object 
#  22  NAME_GOODS_CATEGORY          1670214 non-null  object 
#  23  NAME_PORTFOLIO               1670214 non-null  object 
#  24  NAME_PRODUCT_TYPE            1670214 non-null  object 
#  25  CHANNEL_TYPE                 1670214 non-null  object 
#  26  SELLERPLACE_AREA             1670214 non-null  int64  
#  27  NAME_SELLER_INDUSTRY         1670214 non-null  object 
#  28  CNT_PAYMENT                  1297984 non-null  float64
#  29  NAME_YIELD_GROUP             1670214 non-null  object 
#  30  PRODUCT_COMBINATION          1669868 non-null  object 
#  31  DAYS_FIRST_DRAWING           997149 non-null   float64
#  32  DAYS_FIRST_DUE               997149 non-null   float64
#  33  DAYS_LAST_DUE_1ST_VERSION    997149 non-null   float64
#  34  DAYS_LAST_DUE                997149 non-null   float64
#  35  DAYS_TERMINATION             997149 non-null   float64
#  36  NFLAG_INSURED_ON_APPROVAL    997149 non-null   float64
# dtypes: float64(15), int64(6), object(16)
# memory usage: 471.5+ MB
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 48744 entries, 0 to 48743
# Data columns (total 2 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   SK_ID_CURR  48744 non-null  int64  
#  1   TARGET      48744 non-null  float64
# dtypes: float64(1), int64(1)
# memory usage: 761.8 KB



#====================================================================
# Utilisation du kernel LightGBM de kaggle (proposer par OC)
#====================================================================

#  HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables. 
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

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

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv(data_path_base + 'application_train.csv', nrows= num_rows)
    test_df = pd.read_csv(data_path_base + 'application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv(data_path_base + 'bureau.csv', nrows = num_rows)
    bb = pd.read_csv(data_path_base + 'bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv(data_path_base + 'previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv(data_path_base + 'POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv(data_path_base + 'installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv(data_path_base + 'credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            #nthread=4,
            n_jobs=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

import re
def main(debug = False):
    num_rows = 1000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 10, stratified= False, debug= debug)
    return df, feat_importance



#--------------------------------------------------------------------
# création et sauvegarde du nouveau dataframe « light »  
# et des features importantes identidiées
#--------------------------------------------------------------------

df_light,feature_importance_df = main()

# sauvegarde 
df_light.to_csv(result_path_base + "data_light.csv")
feature_importance_df.to_csv(result_path_base + "df_features_importances.csv")































