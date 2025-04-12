import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Combine for easier processing
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Parameters
IMG_SIZE = 128  # Resize images to 128x128 for ORB

# Convert to grayscale and resize
# def preprocess_images(X):
#     processed = []
#     for img in X:
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
#         processed.append(resized)
#     return np.array(processed)

# X_processed = preprocess_images(X)

def preprocess_images(X):
    processed = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for img in X:
        resized = cv2.resize(img, (64, 64))
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        enhanced_img = cv2.merge((cl, a, b))

        # Fixed two-step conversion here:
        rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)
        final_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        processed.append(final_img)
    return np.array(processed)


X_processed = preprocess_images(X)

# Optional: Visualize preprocessed images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_processed[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Train-test split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Save preprocessed images
np.save("X_train_p.npy", X_train_p)
np.save("X_test_p.npy", X_test_p)
np.save("y_train_p.npy", y_train_p)
np.save("y_test_p.npy", y_test_p)


