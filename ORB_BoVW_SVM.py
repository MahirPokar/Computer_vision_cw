import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load preprocessed images (already done)
X_train_p = np.load("X_train_p.npy")
X_test_p = np.load("X_test_p.npy")
y_train_p = np.load("y_train_p.npy")
y_test_p = np.load("y_test_p.npy")

# ORB detector (already suitable)
orb = cv2.ORB_create(nfeatures=500)

# Collect descriptors from training images (already suitable)
descriptor_list = []
valid_images, valid_labels = [], []

for i in tqdm(range(len(X_train_p))):
    kp, des = orb.detectAndCompute(X_train_p[i], None)
    if des is not None:
        descriptor_list.extend(des)
        valid_images.append(i)
        valid_labels.append(y_train_p[i])

descriptor_stack = np.array(descriptor_list)
print("Total descriptors collected:", descriptor_stack.shape)

# Function to build histograms
def build_histograms(images, orb, kmeans, k):
    histograms = []
    for img in tqdm(images):
        kp, des = orb.detectAndCompute(img, None)
        hist = np.zeros(k)
        if des is not None:
            words = kmeans.predict(des)
            hist, _ = np.histogram(words, bins=np.arange(k + 1))
        histograms.append(hist)
    return np.array(histograms)

# Experiments for various cluster sizes and SVM kernels
cluster_sizes = [200, 300, 500]
svm_kernels = [('linear', 1.0), ('rbf', 1.0)]  # (kernel type, C parameter)

results = []

for k in cluster_sizes:
    print(f"\nTesting with {k} clusters...")
    
    # KMeans clustering clearly
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
    kmeans.fit(descriptor_stack)

    # Build histograms for train/test images clearly
    X_train_hist = build_histograms([X_train_p[i] for i in valid_images], orb, kmeans, k)
    y_train_hist = np.array(valid_labels)
    X_test_hist = build_histograms(X_test_p, orb, kmeans, k)

    # SVM kernel testing
    for kernel_type, C_val in svm_kernels:
        print(f"Testing SVM kernel: {kernel_type}, C={C_val}")
        svm = SVC(kernel=kernel_type, C=C_val, gamma='scale', random_state=42)
        svm.fit(X_train_hist, y_train_hist)
        y_pred = svm.predict(X_test_hist)

        acc = accuracy_score(y_test_p, y_pred)
        print(f"Accuracy (clusters={k}, kernel={kernel_type}): {acc:.4f}")

        # Save results
        results.append({'clusters': k, 'kernel': kernel_type, 'C': C_val, 'accuracy': acc})

# Print summary clearly
print("\n=== Experiment Summary ===")
for res in results:
    print(f"Clusters: {res['clusters']}, Kernel: {res['kernel']}, C: {res['C']}, Accuracy: {res['accuracy']:.4f}")
