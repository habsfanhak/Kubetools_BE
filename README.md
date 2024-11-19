# KubeTools Assessment Backend

This is the Kubetools assessment FastAPI backend file. Includes the scripts for training the model, the api routes for returning, and the helper function for loading the model and making a prediction. A regression model is used, trained on a couple features to identify the target overall rating.

## /predict
Takes an object with an abv value, beerId, brewerId, appearance, aroma, palate, taste, style, reviewText
