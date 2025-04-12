import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Create results directory
os.makedirs("results", exist_ok=True)

# Load your new preprocessed images clearly
X_train_p = np.load("X_train_p.npy")
y_train_p = np.load("y_train_p.npy")
X_test_p = np.load("X_test_p.npy")
y_test_p = np.load("y_test_p.npy")

# ORB descriptor extraction
orb = cv2.ORB_create(nfeatures=500)

descriptor_list = []
valid_images, valid_labels = [], []

print("Extracting ORB descriptors clearly...")
for i in tqdm(range(len(X_train_p))):
    kp, des = orb.detectAndCompute(X_train_p[i], None)
    if des is not None:
        descriptor_list.extend(des)
        valid_images.append(i)
        valid_labels.append(y_train_p[i])

descriptor_stack = np.array(descriptor_list)
print("Total descriptors collected:", descriptor_stack.shape)

# KMeans clustering (BoVW vocabulary)
k = 300  # optimal cluster size identified
print(f"\nClustering descriptors into {k} visual words...")
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
kmeans.fit(descriptor_stack)

# Spatial Pyramid Matching implementation
def spm_histogram(img, orb, kmeans, k):
    h, w = img.shape
    histograms = []

    # Level 0 (whole image)
    kp, des = orb.detectAndCompute(img, None)
    hist_full = np.zeros(k)
    if des is not None:
        words = kmeans.predict(des)
        hist_full, _ = np.histogram(words, bins=np.arange(k + 1))
    histograms.append(hist_full)

    # Level 1 (quadrants)
    quadrants = [
        img[0:h//2, 0:w//2], img[0:h//2, w//2:w],
        img[h//2:h, 0:w//2], img[h//2:h, w//2:w]
    ]

    for q in quadrants:
        kp, des = orb.detectAndCompute(q, None)
        hist_q = np.zeros(k)
        if des is not None:
            words = kmeans.predict(des)
            hist_q, _ = np.histogram(words, bins=np.arange(k + 1))
        histograms.append(hist_q)

    return np.concatenate(histograms)

# Build SPM histograms
print("\nGenerating SPM histograms for training set...")
X_train_spm = [spm_histogram(X_train_p[i], orb, kmeans, k) for i in tqdm(valid_images)]
X_train_spm = np.array(X_train_spm)
y_train_hist = np.array(valid_labels)

print("\nGenerating SPM histograms for test set...")
X_test_spm = [spm_histogram(img, orb, kmeans, k) for img in tqdm(X_test_p)]
X_test_spm = np.array(X_test_spm)

# Train SVM classifier with RBF kernel
print("\nTraining SVM classifier (RBF kernel)...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_spm, y_train_hist)

# Evaluate clearly
print("\nEvaluating clearly...")
y_pred = svm.predict(X_test_spm)
acc = accuracy_score(y_test_p, y_pred)
print(f"\nFinal Test Accuracy (ORB+BoVW+SPM+SVM): {acc:.4f}")

# Generate and save confusion matrix
cm = confusion_matrix(y_test_p, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation='vertical', cmap='Blues')
plt.title("Confusion Matrix (ORB+BoVW+SPM+SVM)")
plt.tight_layout()
plt.savefig("results/confusion_matrix_orb_bovw_spm_svm.png")
plt.show()