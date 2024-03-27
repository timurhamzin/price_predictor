# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('sold_items_history.csv')  # Make sure to replace 'path_to_your_data.csv' with your actual data path

# Display the first few rows of the dataframe
data.head()

# Dropping unnecessary columns and rows with missing target variable
data_cleaned = data.drop(['id', 'status', 'builder_id'], axis=1)
data_cleaned = data_cleaned.dropna(subset=['sold_price'])

# Recollect numerical and categorical features without 'sold_price'
numerical_features = [col for col in data_cleaned.columns if (data_cleaned[col].dtype == 'float64' or data_cleaned[col].dtype == 'int64') and col != 'sold_price']
categorical_features = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])

# Splitting the dataset into training and testing sets
X = data_cleaned.drop('sold_price', axis=1)
y = data_cleaned['sold_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline that includes preprocessing and model
# Set n_jobs=-1 to use all available processors
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

# Training the model
pipeline.fit(X_train, y_train)

# Predicting the prices for the test set
y_pred = pipeline.predict(X_test)

# Model Evaluation
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# Feature Importance Assessment
if 'model' in pipeline.named_steps:
    importances = pipeline.named_steps['model'].feature_importances_
    # Transforming encoded feature names back to original
    feature_names = numerical_features.tolist() + \
                    list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names(categorical_features))
    feature_importance = pd.DataFrame(sorted(zip(importances, feature_names)), columns=['Value','Feature']).sort_values(by="Value", ascending=False)
    print(feature_importance)
