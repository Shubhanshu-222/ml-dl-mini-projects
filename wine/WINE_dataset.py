# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:36:59 2025

@author: ASUS
"""

# Import necessary libraries
from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Wine dataset
wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# 1. Pair Plot
sns.pairplot(wine_df, hue='target', diag_kind='kde', palette='Set2')
plt.suptitle("Pair Plot of Wine Dataset", y=1.02)
plt.show()

# 2. Violin Plot for selected features
selected_features = ['alcohol', 'malic_acid', 'ash']
plt.figure(figsize=(12, 6))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(1, 3, i)
    sns.violinplot(data=wine_df, x='target', y=feature, palette='Set3')
    plt.title(f'Violin Plot: {feature}')
plt.tight_layout()
plt.show()

# 3. Scatter Plot: alcohol vs color_intensity
plt.figure(figsize=(8, 6))
sns.scatterplot(data=wine_df, x='alcohol', y='color_intensity', hue='target', palette='coolwarm', s=50)
plt.title('Scatter Plot: Alcohol vs Color Intensity')
plt.xlabel('Alcohol')
plt.ylabel('Color Intensity')
plt.legend(title='Target')
plt.grid(True)
plt.show()

# 4. Histogram for all features
wine_df.iloc[:, :-1].hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histogram of Features', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
