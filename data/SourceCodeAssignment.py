import numpy as np
import pandas as pd
import gradio as gr
import sklearn as sk

# Read the taxi data
data = pd.read_csv('TaxiData.csv')
print(data.head(6))