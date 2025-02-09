# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:38:27 2025

@author: ASUS
"""

# Statistics

# Import required libraries
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris_data = load_iris()

# Create a DataFrame
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target
iris_df['species'] = iris_df['species'].map({
    0: 'setosa', 
    1: 'versicolor', 
    2: 'virginica'
})

# Perform the required analysis
# 1. Dimension of dataset
dimensions = iris_df.shape
print(f"Dimensions of the dataset: {dimensions}")

# 2. Last 5 elements
last_five = iris_df.tail()
print("\nLast 5 elements of the dataset:")
print(last_five)

# 3. Number of classes
num_classes = iris_df['species'].nunique()
print(f"\nNumber of classes: {num_classes}")

# 4. Number of features and their names
num_features = len(iris_data.feature_names)
feature_names = iris_data.feature_names
print(f"\nNumber of features: {num_features}")
print(f"Feature names: {feature_names}")

# 5. Number of instances per class
instances_per_class = iris_df['species'].value_counts()
print(f"\nNumber of instances per class:")
print(instances_per_class)

# 6. First 5 elements
first_five = iris_df.head()
print("\nFirst 5 elements of the dataset:")
print(first_five)

# 7. Five-point summary
five_point_summary = iris_df.describe()
print("\nFive-point summary:")
print(five_point_summary)
