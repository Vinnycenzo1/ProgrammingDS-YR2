product_data=[['e-book', 2000], ['plants', 6000], ['Pencil', 3000]]
print(product_data)

import pandas as pd
df = pd.read_csv('dirtydata.csv')
print(df.head(10))

#Print information about the data:
print(df.info())
print(df.isna().sum())
print(df.duplicated())

new_df = df.dropna()
print(new_df.to_string())