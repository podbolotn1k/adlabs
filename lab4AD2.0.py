import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder

file_path = r'D:\AD\lab4\auto-mpg.data'
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_csv(file_path, header=None, names=column_names,
                 sep=r'\s+', na_values='?')

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower_median'] = df['horsepower'].fillna(df['horsepower'].median())
df['horsepower_interp'] = df['horsepower'].interpolate()
df['horsepower'] = df['horsepower_interp']
df.drop(['horsepower_median', 'horsepower_interp'], axis=1, inplace=True)

def normalize_df(df, columns):
    return df[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

def standardize_df(df, columns):
    return df[columns].apply(lambda x: (x - x.mean()) / x.std())

normalized_data = normalize_df(df, ['mpg', 'cylinders', 'displacement'])
standardized_data = standardize_df(df, ['mpg', 'cylinders', 'displacement'])

plt.figure(figsize=(8, 5))
sns.histplot(df['mpg'], bins=10, kde=True)
plt.title('Histogram of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(x=df.index, y=df['mpg'])
plt.title('Line Plot of MPG over Index')
plt.xlabel('Index')
plt.ylabel('MPG')
plt.show()

pearson_corr, _ = pearsonr(df['horsepower'], df['mpg'])
spearman_corr, _ = spearmanr(df['horsepower'], df['mpg'])

print("-" * 70)
print(f"Pearson correlation (horsepower vs mpg): {pearson_corr:.4f}")
print(f"Spearman correlation (horsepower vs mpg): {spearman_corr:.4f}")
print("-" * 70)

encoder = OneHotEncoder(sparse_output=False)
origin_encoded = encoder.fit_transform(df[['origin']])
origin_df = pd.DataFrame(origin_encoded, columns=encoder.get_feature_names_out(['origin']))


df_encoded = pd.concat([df.drop('origin', axis=1), origin_df], axis=1)
print(df_encoded.head())

# 1
df_numeric = df.select_dtypes(include='number')

df_numeric.hist(
    bins=15, 
    color='steelblue', 
    edgecolor='black', 
    linewidth=1.0,
    xlabelsize=8, 
    ylabelsize=8, 
    grid=False
)

plt.tight_layout(rect=(0, 0, 1.2, 1.2))
plt.suptitle('Гістограми числових ознак автомобілів', fontsize=14, y=1.02)
plt.show()

# 2
plt.figure(figsize=(10, 6))
corr = df_numeric.corr(numeric_only=True)

sns.heatmap(
    round(corr, 2),
    annot=True,
    cmap="coolwarm",
    fmt='.2f',
    linewidths=0.5
)

plt.title('Теплова карта кореляцій між характеристиками авто', fontsize=14)
plt.tight_layout()
plt.show()

# 3
selected_cols = ['mpg', 'horsepower', 'weight', 'acceleration', 'model_year']
df_pairplot = df[selected_cols].copy()

df_pairplot['year_group'] = pd.cut(df['model_year'], bins=3, labels=["старі", "середні", "нові"])

pp = sns.pairplot(
    df_pairplot,
    hue='year_group',
    palette={'старі': '#FF9999', 'середні': '#FFE888', 'нові': '#88FFAA'},
    plot_kws=dict(edgecolor="black", linewidth=0.5),
    diag_kind='kde'
)

pp.fig.subplots_adjust(top=0.92)
pp.fig.suptitle('Парні графіки: MPG, потужність, вага, прискорення', fontsize=14)
plt.show()
