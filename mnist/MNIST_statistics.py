import idx2numpy

# Load the MNIST dataset (replace the file path as needed)
image_file = 'D:/Projects/10MLProjects/MNISTdataset/train-images.idx3-ubyte'
label_file = 'D:/Projects/10MLProjects/MNISTdataset/train-labels.idx1-ubyte'

# Load the images and labels using idx2numpy
images = idx2numpy.convert_from_file(image_file)
labels = idx2numpy.convert_from_file(label_file)

# 1. Dimension of dataset (images and labels)
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# 2. Last 5 elements (last 5 images and their labels)
print("\nLast 5 elements (images and labels):")
print(images[-5:])
print(labels[-5:])

# 3. Number of classes
classes = set(labels)  # Get unique class labels
print(f"\nNumber of classes: {len(classes)}")

# 4. Number of features and their names (for images, the features are the pixel values)
print(f"\nNumber of features (pixels per image): {images.shape[1] * images.shape[2]}")

# 5. Number of instances per class
instances_per_class = {cls: list(labels).count(cls) for cls in classes}
print("\nNumber of instances per class:")
print(instances_per_class)

# 6. First 5 elements (images and their labels)
print("\nFirst 5 elements (images and labels):")
print(images[:5])
print(labels[:5])

# 7. 5-point summary (for the labels - can be used to describe distribution of classes)
import numpy as np
print("\n5-point summary of label distribution:")
print(np.percentile(labels, [0, 25, 50, 75, 100]))
