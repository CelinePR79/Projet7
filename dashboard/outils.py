#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: celine
"""

import os
from datetime import datetime

# retourne dans dossier le premier nom d'un fichier contenant la substring
# dossier doit finir par '/'
def get_nom_fichier(dossier, substring):
	for fname in os.listdir(dossier):
		if substring in fname:
			return fname

# renommer un fichier dans dossier contenant racine_nom avec la date en plus
# Ex : toto_xxx.png => toto_20220719.png'
# dossier doit finir par '/'
def renommer_fichier(dossier, racine_nom, extension):
	fichier = get_nom_fichier(dossier,racine_nom)
	#nom, extension = os.path.splitext(fichier)
	now = datetime.now()
	fichier_new = racine_nom + '_' + now.strftime("%Y%m%d%H%M%S") + extension 
	os.rename(dossier+fichier, dossier+fichier_new)
	return fichier_new
