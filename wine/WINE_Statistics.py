# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:39:24 2025

@author: ASUS
"""
# Import required libraries
from sklearn.datasets import load_wine
import pandas as pd

# Load the Wine dataset
wine_data = load_wine()

# Create a DataFrame
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target
wine_df['target'] = wine_df['target'].map({
    0: 'class_0', 
    1: 'class_1', 
    2: 'class_2'
})

# 1. Dimension of dataset
dimensions = wine_df.shape
print(f"Dimensions of the dataset: {dimensions}")

# 2. Last 5 elements
last_five = wine_df.tail()
print("\nLast 5 elements of the dataset:")
print(last_five)

# 3. Number of classes
num_classes = wine_df['target'].nunique()
print(f"\nNumber of classes: {num_classes}")

# 4. Number of features and their names
num_features = len(wine_data.feature_names)
feature_names = wine_data.feature_names
print(f"\nNumber of features: {num_features}")
print(f"Feature names: {feature_names}")

# 5. Number of instances per class
instances_per_class = wine_df['target'].value_counts()
print(f"\nNumber of instances per class:")
print(instances_per_class)

# 6. First 5 elements
first_five = wine_df.head()
print("\nFirst 5 elements of the dataset:")
print(first_five)

# 7. Five-point summary
five_point_summary = wine_df.describe()
print("\nFive-point summary:")
print(five_point_summary)
