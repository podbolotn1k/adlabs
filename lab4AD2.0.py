import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

file_path = r'D:\AD\lab4\auto-mpg.data'
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_csv(file_path, header=None, names=column_names, sep=r'\s+', na_values='?')

def handle_missing_data(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df[non_numeric_columns] = df[non_numeric_columns].fillna('Unknown')
handle_missing_data(df)

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def standardize(arr):
    return (arr - arr.mean()) / arr.std()

normalized_data = normalize(df[['mpg', 'cylinders', 'displacement']].values)
standardized_data = standardize(df[['mpg', 'cylinders', 'displacement']].values)

plt.hist(df['mpg'], bins=10, edgecolor='black')
plt.title('Histogram of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df['horsepower'], df['mpg'])
plt.title('Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()

pearson_corr = np.corrcoef(df['horsepower'], df['mpg'])[0, 1]
spearman_corr, _ = spearmanr(df['horsepower'], df['mpg'])
print(f"-"*70)
print(f"Pearson correlation: {pearson_corr}")
print(f"Spearman correlation: {spearman_corr}")
print(f"-"*70)

df_encoded = pd.get_dummies(df, columns=['origin'])
print(df_encoded.head())
