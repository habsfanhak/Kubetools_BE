import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.losses import MeanSquaredError
import joblib  # To save the transformers

# Loading the dataset
data = pd.read_csv("train.csv")

# Removing the NaN reviews with empty strings
data['review/text'] = data['review/text'].fillna('') 

# Defining the features and the target
X = data[['beer/ABV', 'beer/beerId', 'beer/brewerId', 'review/appearance', 'review/aroma', 
          'review/palate', 'review/taste', 'beer/style', 'review/text']]
y = data['review/overall']

# Splits and then further splits for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Listing numeric and categorical features
numeric_columns = ['beer/ABV', 'beer/beerId', 'beer/brewerId', 'review/appearance', 
                   'review/aroma', 'review/palate', 'review/taste']
categorical_columns = ['beer/style']

# Creating transformers for preprocessing
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Preprocessing the data and combines it all for our training data
numeric_data = numeric_transformer.fit_transform(X_train[numeric_columns])
categorical_data = categorical_transformer.fit_transform(X_train[categorical_columns])
text_data = tfidf_vectorizer.fit_transform(X_train['review/text']).toarray()
X_train_processed = np.hstack([numeric_data, categorical_data, text_data])

# Same for the validation data
numeric_data_val = numeric_transformer.transform(X_val[numeric_columns])
categorical_data_val = categorical_transformer.transform(X_val[categorical_columns])
text_data_val = tfidf_vectorizer.transform(X_val['review/text']).toarray()
X_val_processed = np.hstack([numeric_data_val, categorical_data_val, text_data_val])

# Defining the model
input_shape = X_train_processed.shape[1]
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compiling the model
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanSquaredError()])

# Training the model
model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Saving the model and the transformers
model.save("beer_rating.h5")
joblib.dump(numeric_transformer, "numeric_transformer.pkl")
joblib.dump(categorical_transformer, "categorical_transformer.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Evaluation on test data
numeric_data_test = numeric_transformer.transform(X_test[numeric_columns])
categorical_data_test = categorical_transformer.transform(X_test[categorical_columns])
text_data_test = tfidf_vectorizer.transform(X_test['review/text']).toarray()
X_test_processed = np.hstack([numeric_data_test, categorical_data_test, text_data_test])
y_test_pred = model.predict(X_test_processed).flatten()

y_val_pred = model.predict(X_val_processed).flatten()

# Measures how many predictions got within 1 of the actual value
tolerance = 1
correct_predictions = np.abs(y_val - y_val_pred) <= tolerance
percentage_correct = np.mean(correct_predictions) * 100

# Measuring percentage of predictions that were within 1 point of the actual
print(f"Percentage of predictions within ±{tolerance}: {percentage_correct:.2f}%")
