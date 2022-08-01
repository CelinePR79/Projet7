#!/usr/bin/env python
# coding: utf-8

path_data_model = 'Resultats/'
tests_path_base = 'tests/'

import pandas as pd
import joblib

def charger_donnees_et_modele(path_data_model):
    # Chargement des datas
    data = pd.read_csv(path_data_model + 'df_10_features_80.csv', sep=',')
    list_features_only=['PAYMENT_RATE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_ID_PUBLISH','ACTIVE_DAYS_CREDIT_MAX','ACTIVE_DAYS_CREDIT_ENDDATE_MIN']
    data_features = data[list_features_only]
    
    ## chargement du modèle
    X_train = pd.read_csv(path_data_model + 'X_train.csv')
    y_train = pd.read_csv(path_data_model + 'y_train.csv')
    X_test  = pd.read_csv(path_data_model + 'X_test.csv')
    y_test  = pd.read_csv(path_data_model + 'y_test.csv')
    
    model = joblib.load(path_data_model + 'pipeline_credit.joblib') 
    #clf = model["clf"]
    #scaler = model["scaler"]
    
    return data, data_features, model, X_train, y_train, X_test, y_test

data, data_features, model, X_train, y_train, X_test, y_test = charger_donnees_et_modele(path_data_model)

# vérification sur un individu
#model.predict(X_test.sample())  # 0 <=> rembourement OK ;  1  <=> rembourement KO


#=========================================================================
# Explanations Globales
#=========================================================================

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## Explication globale

# For Tree Models
def model_gobal_explanation_as_figure(data, classifier, top_x, figure_fullpath, title='', color_palette_name='Paired'):
    feature_importance = classifier.feature_importances_
    indices = np.argsort(feature_importance)
    indices = indices[-top_x:]
        
    if color_palette_name:
        color_list =  sns.color_palette(color_palette_name, len(data.columns)) # dark | partel | hls | Paired | Reds | ...
    
    fig = plt.figure()
    #fig.adjust(hspace = 0.5, wspace=0.8)
    bars = plt.barh(range(len(indices)), feature_importance[indices], color='b', align='center') 
    plt.yticks(range(len(indices)), [data.columns[j] for j in indices], fontweight="normal", fontsize=16) 
    for i, ticklabel in enumerate(plt.gca().get_yticklabels()):
        ticklabel.set_color(color_list[indices[i]])  
    for i,bar in enumerate(bars):
        bar.set_color(color_list[indices[i]]) 
    
    if title:
        plt.title(title, figure=fig)
    
    fig.savefig(figure_fullpath, bbox_inches="tight") # bbox_inches pour ne pas tronquer les légendes
    
    return fig

top_x = 10 # number of x most important features to show
titre = 'Feature Importance for the Random Forest Models. Top ' + str(top_x) + ' Features.'
global_fig_path = path_data_model + 'global_shap_fig.png'

model_gobal_explanation_as_figure(data, model["clf"], top_x, global_fig_path, titre)



#=========================================================================
# Explanations d'un individu
#=========================================================================

import numpy as np

def individual_random_index(X_test):
    sample_index = np.random.randint(X_test.shape[0])
    return sample_index

def get_individual_by_index(X_test, index):
    individual = X_test.iloc[[index]]
    return individual

def client_remboursement_ok(client, model):
    prediction = model.predict(client)
    return prediction[0] == 0  # 0 <=> crédit remboursé

def client_remboursement_proba(client, model):
    score = model.predict_proba(client)
    return score[0][0]

def client_remboursement_proba(client, model):
    score = model.predict_proba(client)
    return score[0][0]


import plotly.graph_objects as go
# pip install -U plotly
# pip install -U kaleido # pour enregistrement
def client_remboursement_proba_as_figure(proba, img_path):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Proba de remboursement"}))
    fig.write_image(img_path)
    plt.close() # do not display figure
   
 
sample_index = individual_random_index(X_test)

client = get_individual_by_index(X_test, sample_index)

client_remboursement = client_remboursement_ok(client, model)

client_proba_remb = client_remboursement_proba(client, model)

client_image_proba = 'proba.png'
client_remboursement_proba_as_figure(client_proba_remb, tests_path_base+client_image_proba)



#-------------------------------------------------------------------------
# Distribution selon classe
#-------------------------------------------------------------------------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# distrition de la feature et affcihage de l'individu
def distribution_plot (feature,data,y_test,individu_index, img_path):
    #plt.hist(x=feature, data=data[data.index.isin(y_test.index)])
    #plt.clf()
    _ = sns.displot(x=feature,data=data[data.index.isin(y_test.index)],kind='kde',hue='TARGET',fill=True,lw=3)
    plt.axvline(x=data[data.index==individu_index][feature].values,color='red',label='individu')
    plt.title('Distribution')
    plt.legend()
    plt.savefig(img_path, bbox_inches="tight")
    plt.close() # do not display figure

var='DAYS_BIRTH'
individu=np.random.randint(X_test.shape[0])   
distribution_plot(var,data,y_test,individu, tests_path_base+'distribution_plot.png')




#-------------------------------------------------------------------------
# Analyse bi-variées
#-------------------------------------------------------------------------
import numpy as np
import seaborn as sns

#data_features
def scatter_plot(X_test,model,y_test,var1,var2,individual_index, img_path):
    #Probabilité de tout l'echantillon
    prediction_total=model.predict_proba(X_test)
    #création du DataFrame associé
    data_prediction = pd.DataFrame(prediction_total,columns=['prob_positif','proba_negatif'])
    # X_test + 1 colonne 'prediction'
    X_test_with_prediction = X_test.copy()
    X_test_with_prediction['prediction'] = data_prediction['prob_positif']
    # scatter plot
    
    f = sns.scatterplot(x=var1,y=var2,data=X_test_with_prediction,hue='prediction')
    sns.scatterplot(x=var1,y=var2,data=X_test_with_prediction.iloc[[individual_index]],color='red',label='individu')
    fig = f.get_figure()
    fig.savefig(img_path, bbox_inches="tight")
    plt.close() # do not display figure

var1='EXT_SOURCE_3'
var2='PAYMENT_RATE'
indiv_index = 4
scatter_plot(X_test,model,y_test,var1,var2,indiv_index, tests_path_base+'scatter_plot.png')





import shap
# SYNTAXE :
#explainer = shap.KernelExplainer(model.predict_proba, background_data)
#shap_values = explainer.shap_values(X_test)
#   model
#       the model to be explained
#   background_data 
#       This is a required argument to KernelExplainer. 
#       Since most models aren’t designed to handle arbitrary missing data at test time, SHAP simulates a “missing” feature by replacing it with the values it takes in the background dataset. 
#       For small problems, this background dataset can be the whole training set, but for larger problems, it is suggested that a subsample of the training set (or the kmeans function to summarize the dataset) is used. 
#       Background data is optional for tree-based models.
#   explainer.expected_value
#       This is a field in the explainer object is displayed as the baseline in a SHAP force plot. It should be the same as the mean of the model output over the background dataset. One simple task which I found to be useful is to manually compute the mean prediction on the background dataset and see how it corresponds to the expected value output by SHAP.
#   shap_values
#       The shap_values returned by the explainer object are a measure of how each feature contributes to the difference between the model’s expected value and the prediction for that instance. The units of the Shapley values are in the units of the target variable. The sum of the shap values should be equal to the difference between the base value and the model prediction.  


def individual_explanation_as_figure(model, X_train, X_test, individual_index, figure_fullpath):
    if not individual_index in range (0,X_test.shape[0]):
        return None
    
    clf = model["clf"]
    scaler = model["scaler"]
	
    scaled_train_data = scaler.transform(X_train)
    scaled_test_data = scaler.transform(X_test) 
    subsampled_test_data =scaled_test_data[individual_index].reshape(1,-1)
    
    #start_time = time.time()  # import time
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(subsampled_test_data) # Rappel : Background data is optional for tree-based models.
    #elapsed_time = time.time() - start_time

    #print("Tree Explainer SHAP run time", round(elapsed_time,3) , " seconds. ")
    #print("SHAP expected value", explainer.expected_value)
    #print("Model mean value", clf.predict_proba(scaled_train_data).mean(axis=0))
    #print("Model prediction for test data", clf.predict_proba(subsampled_test_data))
    shap.initjs()
    pred_ind = 0
    
    fig = shap.force_plot(explainer.expected_value[1], shap_values[1][0], subsampled_test_data[0], feature_names=X_train.columns, matplotlib=True, show=False) 
    fig.savefig( figure_fullpath, bbox_inches="tight" ) # utilisation de savefig si option show=False dans force_plot
    #shap.save_html(path_data_model+'explainer.html', fig)
    plt.close() # do not display figure

indiv_fig_path = tests_path_base + 'indiv_shap_fig.png'
individual_explanation_as_figure(model, X_train,X_test, sample_index, indiv_fig_path)

