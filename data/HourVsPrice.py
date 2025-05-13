import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('TaxiData.csv')

print(df.head())

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Remove rows where 'price' is NaN
df = df.dropna(subset=['price'])

# Convert 'price' to numeric, forcing errors to NaN
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Drop any rows where 'price' is still NaN after conversion
df = df.dropna(subset=['price'])

# Group by the 'hour' column and calculate average price
hourly_avg = df.groupby('hour')['price'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 5))
sns.lineplot(data=hourly_avg, x='hour', y='price', marker='o', color='blue')

plt.title('Average Taxi Price by Hour of the Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Price ($)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()