# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:36:46 2025

@author: ASUS
"""

# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(
    data=iris_data.data, 
    columns=iris_data.feature_names
)
iris_df['species'] = iris_data.target
iris_df['species'] = iris_df['species'].map({
    0: 'setosa', 
    1: 'versicolor', 
    2: 'virginica'
})

# Visualization

# Pair Plot
sns.pairplot(iris_df, hue='species', diag_kind='kde')
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal length (cm)', data=iris_df, palette='muted')
plt.title("Violin Plot: Sepal Length by Species")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='sepal length (cm)', 
    y='sepal width (cm)', 
    hue='species', 
    data=iris_df, 
    palette='viridis'
)
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(data=iris_df, x='petal length (cm)', hue='species', kde=True, palette='Set2', bins=15)
plt.title("Histogram: Petal Length Distribution by Species")
plt.show()

