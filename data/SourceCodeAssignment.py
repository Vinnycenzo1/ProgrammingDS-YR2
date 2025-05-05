import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import gradio as gr
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

# Read the taxi data
data = pd.read_csv('TaxiData.csv')
print(data.head(6))
print(data.info())

# As there are a few '?' in my data we need to remove them
data.replace('?', np.nan, inplace=True)

# Remove rows where 'datetime' is NA
data = data[data['datetime'] != 'NA']

# Parse the datetime column
data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')

# Remove any rows with missing data
data = data.dropna(subset=['price', 'distance', 'temperature', 'windSpeed', 'visibility'])

# This extracts time features
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek

# Features and Targets
target = 'price'
numeric_features = ['distance', 'temperature', 'windSpeed', 'visibility', 'hour', 'day_of_week']
categorical_features = ['cab_type', 'name', 'short_summary']

numeric_features += ['windBearing', 'humidity']
categorical_features += ['source', 'destination']

# Convert windBearing and humidity to numeric, handle '?' etc.
data['windBearing'] = pd.to_numeric(data['windBearing'], errors='coerce')
data['humidity'] = pd.to_numeric(data['humidity'], errors='coerce')

# Replace '?' or any invalid values (if needed)
data.replace('?', np.nan, inplace=True)

# Drop rows with missing values in new features
data = data.dropna(subset=numeric_features + categorical_features)

for col in numeric_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Removes any remaining NAs
data = data.dropna(subset=numeric_features + categorical_features)

# Define features and target
X = data[numeric_features + categorical_features]
y = data[target]

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # keep numeric features
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = pipeline.score(X_test, y_test)

# Use 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Get R^2 scores across 5 folds
cv_r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')

# Get negative MSE (so it takes negative to make it positive)
cv_neg_mse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_neg_mse_scores)

# Evaluate Random Forest
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rf_r2_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='r2')
rf_neg_mse_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
rf_rmse_scores = np.sqrt(-rf_neg_mse_scores)

print(f'RMSE: {rmse:.2f}')
print(f'R² Score: {r2:.3f}')

print("Cross-validated R² scores:", cv_r2_scores)
print(f"Average R²: {np.mean(cv_r2_scores):.3f}")

print("Cross-validated RMSE scores:", cv_rmse_scores)
print(f"Average RMSE: {np.mean(cv_rmse_scores):.2f}")

print("Random Forest - Cross-validated R² scores:", rf_r2_scores)
print(f"Random Forest - Average R²: {np.mean(rf_r2_scores):.3f}")

print("Random Forest - Cross-validated RMSE scores:", rf_rmse_scores)
print(f"Random Forest - Average RMSE: {np.mean(rf_rmse_scores):.2f}")