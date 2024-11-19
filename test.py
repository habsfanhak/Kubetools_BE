import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.losses import MeanSquaredError


def predict(abv, beerId, brewerId, appearance, aroma, palate, taste, style, reviewText):
    # Load the saved model
    model = load_model("beer_rating.h5", custom_objects={'MeanSquaredError': MeanSquaredError})

    # Load the transformers
    numeric_transformer = joblib.load("numeric_transformer.pkl")
    categorical_transformer = joblib.load("categorical_transformer.pkl") 
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Sample data from API call
    new_data = {
        'beer/ABV': [abv],
        'beer/beerId': [beerId],
        'beer/brewerId': [brewerId],
        'review/appearance': [appearance],
        'review/aroma': [aroma],
        'review/palate': [palate],
        'review/taste': [taste],
        'beer/style': [style],  
        'review/text': [reviewText]
    }

    # Converting new data into a DataFrame for preprocessing
    new_data_df = pd.DataFrame(new_data)

    # Preprocessing 
    numeric_data_new = numeric_transformer.transform(new_data_df[['beer/ABV', 'beer/beerId', 'beer/brewerId', 
                                                                'review/appearance', 'review/aroma', 
                                                                'review/palate', 'review/taste']])
    categorical_data_new = categorical_transformer.transform(new_data_df[['beer/style']])
    text_data_new = tfidf_vectorizer.transform(new_data_df['review/text']).toarray()

    # Combining all of the processed data
    X_new_processed = np.hstack([numeric_data_new, categorical_data_new, text_data_new])

    # Making prediction on new data
    prediction = model.predict(X_new_processed).flatten()
    
    # Returning the prediction
    return prediction[0]
