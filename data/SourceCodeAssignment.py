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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Read the taxi data
data = pd.read_csv('TaxiData.csv')
print(data.head(6))
print(data.info())

# As there are a few '?' in the data we need to remove them
data.replace('?', np.nan, inplace=True)

# Remove rows where 'datetime' is NA
data = data[data['datetime'] != 'NA']

# Remove any rows with missing data
data = data.dropna(subset=['price', 'distance', 'temperature', 'windSpeed', 'visibility'])

# Parse the datetime column
data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')

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

# Replace '?' or any invalid values
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

# Linear Regression pipeline
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

# SVR Model
# Updated preprocessor with scaling for numeric features
preprocessor_svr = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Support Vector Regression pipeline
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_svr),
    ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.2))
])

# Train-test split (already done, reused)
# X_train, X_test, y_train, y_test

# Fit the pipeline
svr_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred_svr = svr_pipeline.predict(X_test)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

# Cross-validated evaluation
cv_r2_svr = cross_val_score(svr_pipeline, X, y, cv=cv, scoring='r2')
cv_rmse_svr = np.sqrt(-cross_val_score(svr_pipeline, X, y, cv=cv, scoring='neg_mean_squared_error'))

# Output results
print(f"SVR RMSE: {rmse_svr:.2f}")
print(f"SVR R² Score: {r2_svr:.3f}")
print("SVR Cross-validated R² scores:", cv_r2_svr)
print(f"SVR Average R²: {np.mean(cv_r2_svr):.3f}")
print("SVR Cross-validated RMSE scores:", cv_rmse_svr)
print(f"SVR Average RMSE: {np.mean(cv_rmse_svr):.2f}")


# Comparison results table for all 3 models
comparison_data = {
    "Model": ["Linear Regression", "Random Forest", "SVR (RBF Kernel)"],
    "CV Avg R²": [np.mean(cv_r2_scores), np.mean(rf_r2_scores), np.mean(cv_r2_svr)],
    "CV Avg RMSE": [np.mean(cv_rmse_scores), np.mean(rf_rmse_scores), np.mean(cv_rmse_svr)]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)

# Visualisation Graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Plot R² comparison
plt.figure(figsize=(8, 4))
sns.barplot(data=comparison_df, x="Model", y="CV Avg R²", palette="Blues_d")
plt.title("Cross-Validated R² by Model")
plt.ylim(0, 1)
plt.ylabel("R² Score")
plt.grid(True)
plt.show()

# Plot RMSE comparison
plt.figure(figsize=(8, 4))
sns.barplot(data=comparison_df, x="Model", y="CV Avg RMSE", palette="Reds_d")
plt.title("Cross-Validated RMSE by Model")
plt.ylabel("RMSE ($)")
plt.grid(True)
plt.show()

# GUI Creation
def predict_fare(hour, source, destination, cab_type, name, distance, temperature, short_summary, humidity, windSpeed, visibility, windBearing):
    input_df = pd.DataFrame([{
        'hour': hour,
        'source': source,
        'destination': destination,
        'cab_type': cab_type,
        'name': name,
        'distance': distance,
        'temperature': temperature,
        'short_summary': short_summary,
        'humidity': humidity,
        'windSpeed': windSpeed,
        'visibility': visibility,
        'windBearing': windBearing

    }])

    return rf_pipeline.predict(input_df)[0]  # or use your best model

# Drop down options
cab_types = list(data['cab_type'].unique())
names = list(data['name'].unique())
summaries = list(data['short_summary'].unique())
sources = list(data['source'].unique())
destinations = list(data['destination'].unique())

inputs = [
    gr.Slider(0, 23, step=1, label="Hour of Day"),
    gr.Dropdown(sources, label="Pickup Location"),
    gr.Dropdown(destinations, label="Drop-off Location"),
    gr.Dropdown(cab_types, label="Cab Type"),
    gr.Dropdown(names, label="Cab Name"),
    gr.Number(label="Distance (km)"),
    gr.Number(label="Temperature (°F)"),
    gr.Dropdown(summaries, label="Weather Summary"),
    gr.Number(label="Humidity (0-1)"),
    gr.Number(label="Wind Speed (km/h)"),
    gr.Number(label="Visibility (km)"),
    gr.Number(label="Wind Bearing (°)")
]

output = gr.Number(label="Predicted Price ($)")

app = gr.Interface(fn=predict_fare, inputs=inputs, outputs=output, title="Taxi Fare Predictor")

app.launch()