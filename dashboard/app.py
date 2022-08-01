#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: celine
"""

from flask import Flask, render_template, url_for, request, send_from_directory

# pip install -U flask-cors
from flask_cors import CORS # pour éviter que le client reçoive du serveur un message de la forme « blocked by CORS policy: No 'Access-Control-Allow-Origin' header. »

# NON
#from flask_caching import Cache
#cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})

#--------------------------------------------------------------------
# Conctruction de l'objet Application
#--------------------------------------------------------------------
app = Flask(__name__)

CORS(app)
#cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

#Bootstrap(app)



#--------------------------------------------------------------------
# Configuration
#--------------------------------------------------------------------

app.config['FLASK_ENV'] = 'production' # 'production' | 'development'
app.config['TESTING'] = False
app.config['DEBUG'] = False
#app.config['SECRET_KEY'] = 'CFLef56NFi'

app.config['STATIC_FOLDER'] = 'static'
app.config['TEMPLATES_FOLDER'] = 'templates'

## ou bien
# app.config.from_pyfile('config.py')


# Supprimer le Cache dans le navigateur. Ne fonctonne pas avec les images ?!?
#@app.after_request
#def add_header(r):
    #r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    #r.headers["Pragma"] = "no-cache"
    #r.headers["Expires"] = "0"
    #r.headers['Cache-Control'] = 'public, max-age=0'
    #return r


#--------------------------------------------------------------------
# Les variables globales, au lancement du programme
#--------------------------------------------------------------------

URL_base = 'http://localhost:5000/'

import prediction_credit as credit

path_data_model = 'data_and_model/'
data, data_features, model, X_train, y_train, X_test, y_test = credit.charger_donnees_et_modele(path_data_model)
#clf = model["clf"]
#scaler = model["scaler"]

# Remarque : dans 'static/' l'image 'global_shap_fig.png' correspond à l'explication global du modèle selon les features.

img_path_base = 'result_images/'


#--------------------------------------------------------------------
# API 
#--------------------------------------------------------------------

import outils

@app.route('/api_proba_remboursement')
def api_proba_remboursement():
	client_index = request.args.get('index_client')
	client = credit.get_individual_by_index(X_test, client_index)
	return credit.client_remboursement_proba(client, model)

@app.route('/api_update_distribution_image')
def api_update_distribution_image():
	# récupération des variables de la requête GET
	client_index = int( request.args.get('index_client') )
	feature = request.args.get('feature')
	# MAJ de l'image de la distribution
	#client_distribution_image = 'client_distribution.png'
	client_distribution_image = outils.renommer_fichier(img_path_base, 'client_distribution', '.png') # renommage pour cachebreaker
	img_path = img_path_base+client_distribution_image
	credit.distribution_plot (feature,data,y_test,client_index, img_path)
	return img_path

@app.route('/api_update_scatterplot_image')
def api_update_scatterplot_image():
	# récupération des variables de la requête GET
	client_index = int( request.args.get('index_client') )
	var1 = request.args.get('var1')
	var2 = request.args.get('var2')
	# MAJ de l'image scatter plot
	#client_scatterplot_img = 'client_scatterplot.png'
	client_scatterplot_img = outils.renommer_fichier(img_path_base, 'client_scatterplot', '.png') # renommage pour cachebreaker
	img_path = img_path_base+client_scatterplot_img
	credit.scatter_plot(X_test,model,y_test,var1,var2,client_index, img_path)
	return img_path


#--------------------------------------------------------------------
# Pages web, dont le Dashboard
#--------------------------------------------------------------------

#@app.route('/', methods=['GET', 'POST'])
@app.route('/') 
@app.route('/index')
def index():
	return render_template('index.html', url_app=URL_base)  # url_app pour le menu

@app.route('/credit')
def creditt():   # Remarque (incompréhensible) : erreur si 'credit'
	return render_template('credit.html', url_app=URL_base)  # url_app pour le menu


#@app.route('/result') 
@app.route('/dashboard')  # chemin dans l'URL où répondre (par défaut http://127.0.0.1:5000/toto)
def dashboard():
	arg_index = request.args.get('index_client') # type str
	
	client_index = 0 # type int
	if arg_index:
		client_index = int(arg_index)
	else:
		client_index = credit.individual_random_index(X_test)

	client = credit.get_individual_by_index(X_test, client_index)

	client_remboursement = credit.client_remboursement_ok(client, model)

	client_proba_remb = credit.client_remboursement_proba(client, model)
	
	#client_proba_image = 'client_proba.png'
	client_proba_image = outils.renommer_fichier(img_path_base, 'client_proba', '.png') # renommage pour cachebreaker
	client_proba_titre= 'Proba de remboursement'
	credit.client_remboursement_proba_as_figure(client_proba_remb, img_path_base+client_proba_image, client_proba_titre)
	
	#client_distribution_image = 'client_distribution.png'
	client_distribution_image = outils.renommer_fichier(img_path_base, 'client_distribution', '.png') # renommage pour cachebreaker
	feature = 'EXT_SOURCE_3'
	credit.distribution_plot (feature,data,y_test,client_index, img_path_base+client_distribution_image)
	
	#client_scatterplot_img = 'client_scatterplot.png'
	client_scatterplot_img = outils.renommer_fichier(img_path_base, 'client_scatterplot', '.png') # renommage pour cachebreaker
	var1='EXT_SOURCE_3'
	var2='EXT_SOURCE_1'
	credit.scatter_plot(X_test,model,y_test,var1,var2,client_index, img_path_base+client_scatterplot_img)
	
	#client_shap_image = 'client_shap.png'
	client_shap_image = outils.renommer_fichier(img_path_base, 'client_shap', '.png') # renommage pour cachebreaker
	credit.individual_explanation_as_figure(model, X_train,X_test, client_index, img_path_base+client_shap_image)
	
	return render_template('dashboard.html', 
						client_index=client_index, 
						remboursement = client_remboursement, 
						proba = client_proba_remb, 
						image_proba = client_proba_image, 
						image_distribution = client_distribution_image,
						image_scatterplot = client_scatterplot_img,
						image_shap = client_shap_image,
						url_app=URL_base # pour le menu
						)


# pour accéder aux images dynamiquement crées lors du résultat (donc pas dans le dossier 'static')
@app.route("/result_images/<filename>")
def result_images(filename):
	#return send_from_directory(app.config["MEDIA_FOLDER], filename, as_attachment=True)
	return send_from_directory('result_images', filename)


if __name__ == "__main__":
    app.run()
    #app.run(threaded=False)  # threaded=False supprime les warning "Starting a Matplotlib GUI outside of the main thread will likely fail."
