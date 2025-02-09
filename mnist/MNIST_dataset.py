# Import required libraries
import idx2numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load MNIST subset data (train-images.idx3-ubyte)
file_path = "D:/Projects/10MLProjects/MNISTdataset/train-images.idx3-ubyte"
images = idx2numpy.convert_from_file(file_path)  # Shape: (num_images, 28, 28)

# Load labels (train-labels.idx1-ubyte, if available)
label_file_path = "D:/Projects/10MLProjects/MNISTdataset/train-labels.idx1-ubyte"
labels = idx2numpy.convert_from_file(label_file_path)  # Shape: (num_images,)

# Reshape images to a 2D array for PCA (28x28 pixels to a single vector of 784 features per image)
X = images.reshape(images.shape[0], -1)  # Shape: (num_images, 784)

# Create a DataFrame for PCA processing
y = pd.Series(labels, name='target')

# Apply PCA for dimensionality reduction (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pd.DataFrame(pca.fit_transform(X), columns=['PCA1', 'PCA2'])
X_pca['target'] = y.astype(int)

# Use a sample of 500 rows to simplify plotting
sample_data = X_pca.sample(500, random_state=42)

# 1. Pair Plot
sns.pairplot(sample_data, hue='target', diag_kind='kde', palette='Set2')
plt.suptitle("Pair Plot of MNIST Dataset (PCA-reduced)", y=1.02)
plt.show()

# 2. Violin Plot for PCA1 and PCA2
plt.figure(figsize=(12, 6))
sns.violinplot(data=sample_data, x='target', y='PCA1', palette='Set3')
plt.title("Violin Plot for PCA1")
plt.show()

sns.violinplot(data=sample_data, x='target', y='PCA2', palette='Set3')
plt.title("Violin Plot for PCA2")
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=sample_data, x='PCA1', y='PCA2', hue='target', palette='tab10', s=50)
plt.title("Scatter Plot of PCA1 vs PCA2")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# 4. Histogram for PCA1 and PCA2
plt.figure(figsize=(12, 6))
plt.hist(sample_data['PCA1'], bins=20, alpha=0.7, label='PCA1', color='blue', edgecolor='black')
plt.hist(sample_data['PCA2'], bins=20, alpha=0.7, label='PCA2', color='orange', edgecolor='black')
plt.title("Histogram of PCA1 and PCA2")
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
