//-------------------------------------------------------------------
// Variables globales
//-------------------------------------------------------------------

var API_BASE = 'http://127.0.0.1:5000/';

//-------------------------------------------------------------------
// Pour mises à jour des figures
//-------------------------------------------------------------------

function distribution_change() {
	var client_index = document.getElementById('client_index');
	var image = document.getElementById('image_distribution');
	var selection = document.getElementById('feature_for_distribution');
	
	// configuration de la requête
	var request = new XMLHttpRequest();
	var url = API_BASE + 'api_update_distribution_image';
		url += '?index_client=' + client_index.innerText;
		url += '&feature=' + selection.value;
	
	request.open('GET', url);
	request.responseType = 'text';
	request.onload = function() {
		if (request.status === 200) { // OK
			reponse = request.responseText;
			image.src = reponse;
			//alert('ici ' + reponse);
		} 
		else {
			console.log('URL not valid or access failed (' + url + ')');
		}
	};
	// envoi de la requête
	request.send();
}

function scatterplot_change() {
	var client_index = document.getElementById('client_index');
	var image = document.getElementById('image_scatterplot');
	var selection1 = document.getElementById('var1_for_scatterplot');
	var selection2 = document.getElementById('var2_for_scatterplot');
	
	// configuration de la requête
	var request = new XMLHttpRequest();
	var url = API_BASE + 'api_update_scatterplot_image';
		url += '?index_client=' + client_index.innerText;
		url += '&var1=' + selection1.value;
		url += '&var2=' + selection2.value;
	
	request.open('GET', url);
	request.responseType = 'text';
	request.onload = function() {
		if (request.status === 200) { // OK
			reponse = request.responseText;
			image.src = reponse;
		} 
		else {
			console.log('URL not valid or access failed (' + url + ')');
		}
	};
	// envoi de la requête
	request.send();
}
