import os
import requests
import sqlite3
import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Function to download the database file
def download_db(url, local_filename):
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Download the databases
download_db('https://techassessment.blob.core.windows.net/aiap18-assessment-data/weather.db', 'data/weather.db')
download_db('https://techassessment.blob.core.windows.net/aiap18-assessment-data/air_quality.db', 'data/air_quality.db')

# Function to load data from SQLite databases
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# Load the datasets
weather_data = load_data('data/weather.db', 'weather')
air_quality_data = load_data('data/air_quality.db', 'air_quality')

# Merge datasets
df = pd.merge(weather_data, air_quality_data, on='data_ref', how='inner')

# Drop unnecessary columns
columns_to_drop = ['data_ref', 'date_x', 'date_y']
df = df.drop(columns=columns_to_drop)

# Custom Transformer for Feature Engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Clean and convert columns to numeric
        def clean_and_convert(df, columns):
            def replace_non_numeric(val):
                if isinstance(val, str):
                    return re.sub(r'[^0-9.-]', '', val)
                return val

            for column in columns:
                df[column] = df[column].apply(replace_non_numeric)
                df[column] = pd.to_numeric(df[column], errors='coerce')
            return df

        # List of columns to clean and convert
        columns_to_convert = [
            'Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)', 
            'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)', 
            'Min Temperature (deg C)', 'Maximum Temperature (deg C)', 
            'Min Wind Speed (km/h)', 'Max Wind Speed (km/h)', 
            'Sunshine Duration (hrs)', 'Cloud Cover (%)',
            'pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 
            'pm25_central', 'psi_north', 'psi_south', 'psi_east', 
            'psi_west', 'psi_central'
        ]

        # Clean and convert the specified columns
        df = clean_and_convert(X, columns_to_convert)

        # Replace NaN values with the median of the respective columns
        df[columns_to_convert] = df[columns_to_convert].fillna(df[columns_to_convert].median())

        # Handle 'Dew Point Category'
        df['Dew Point Category'] = df['Dew Point Category'].str.upper()
        dew_point_mapping = {
            'H': 'HIGH',
            'VH': 'VERY HIGH',
            'M': 'MODERATE',
            'L': 'LOW',
            'VL': 'VERY LOW',
            'MINIMAL': 'VERY LOW',
            'NORMAL': 'MODERATE',
            'EXTREME': 'VERY HIGH',
            'BELOW AVERAGE': 'LOW'
        }
        df['Dew Point Category'] = df['Dew Point Category'].replace(dew_point_mapping)

        # Handle 'Wind Direction'
        df['Wind Direction'] = df['Wind Direction'].str.rstrip('.').str.upper()
        wind_direction_mapping = {
            'SOUTHEAST': 'SE',
            'SOUTH': 'S',
            'SOUTHWARD': 'S',
            'SW': 'SW',
            'NORTH': 'N',
            'NORTHWARD': 'N',
            'NE': 'NE',
            'NORTHEAST': 'NE',
            'NW': 'NW',
            'NORTHWEST': 'NW',
            'WEST': 'W',
            'EAST': 'E',
            'E': 'E',
            'W': 'W'
        }
        df['Wind Direction'] = df['Wind Direction'].replace(wind_direction_mapping)

        # Combine PM2.5 columns into one
        df['PM25 Combined'] = df[['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central']].mean(axis=1)

        # Combine PSI columns into one
        df['PSI Combined'] = df[['psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']].mean(axis=1)

        # Drop the original individual columns
        df = df.drop(columns=['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central', 
                              'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central'])

        return df

# Define feature types for preprocessing
numeric_features = [
    'Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)', 
    'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)', 
    'Min Temperature (deg C)', 'Maximum Temperature (deg C)', 
    'Min Wind Speed (km/h)', 'Max Wind Speed (km/h)', 
    'Sunshine Duration (hrs)', 'Cloud Cover (%)',
    'Wet Bulb Temperature (deg F)', 'Relative Humidity (%)', 
    'Air Pressure (hPa)', 'PM25 Combined', 'PSI Combined'
]
categorical_features = ['Dew Point Category', 'Wind Direction']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the pipeline with feature engineering step
pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineering()),  # Add custom feature engineering step
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data into features and target
X = df.drop('Daily Solar Panel Efficiency', axis=1)
y = df['Daily Solar Panel Efficiency']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

for model_name, model in models.items():
    pipeline.set_params(classifier=model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"{model_name} Model")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(pipeline, f'{model_name.lower().replace(" ", "_")}_pipeline.joblib')

param_grids = {
    'Random Forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    }
}

# Perform grid search for each model
best_estimators = {}
for model_name, param_grid in param_grids.items():
    print(f"Fine-tuning {model_name}...")
    
    if model_name == 'Random Forest':
        pipeline.set_params(classifier=RandomForestClassifier(random_state=42))
    elif model_name == 'Gradient Boosting':
        pipeline.set_params(classifier=GradientBoostingClassifier(random_state=42))
    elif model_name == 'Logistic Regression':
        pipeline.set_params(classifier=LogisticRegression(random_state=42, max_iter=1000))
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimators[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    print(f"{model_name} Model After Fine-Tuning")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(grid_search.best_estimator_, f'{model_name.lower().replace(" ", "_")}_best_pipeline.joblib')