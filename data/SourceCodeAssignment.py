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


# Read the taxi data
data = pd.read_csv('TaxiData.csv')
print(data.head(6))

# Remove rows where 'datetime' is NA
data = data[data['datetime'] != 'NA']

# Parse the datetime column
data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')

# Remove any rows with missing data
data = data.dropna()

# This extracts time features
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek

# Features and Targets
target = 'price'
numeric_features = ['distance', 'temperature', 'windSpeed', 'visibility', 'hour', 'day_of_week']
categorical_features = ['cab_type', 'name', 'short_summary']