import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
# Load preprocessed images
X_train_p = np.load("X_train_p.npy")
X_test_p = np.load("X_test_p.npy")
y_train_p = np.load("y_train_p.npy")
y_test_p = np.load("y_test_p.npy")

# ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Collect descriptors from training images
descriptor_list = []
valid_images = []  # To track which images had descriptors
valid_labels = []

for i in tqdm(range(len(X_train_p))):
    kp, des = orb.detectAndCompute(X_train_p[i], None)
    if des is not None:
        descriptor_list.extend(des)
        valid_images.append(i)
        valid_labels.append(y_train_p[i])

descriptor_stack = np.array(descriptor_list)

print("Total descriptors collected:", descriptor_stack.shape)

# === KMeans clustering to build vocabulary ===
k = 100  # Number of visual words
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
kmeans.fit(descriptor_stack)

# === Build histograms for each image ===
def build_histograms(images, orb, kmeans):
    histograms = []
    for img in tqdm(images):
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            words = kmeans.predict(des)
            hist, _ = np.histogram(words, bins=np.arange(k + 1))
        else:
            hist = np.zeros(k)
        histograms.append(hist)
    return np.array(histograms)

X_train_hist = build_histograms([X_train_p[i] for i in valid_images], orb, kmeans)
y_train_hist = np.array(valid_labels)

X_test_hist = build_histograms(X_test_p, orb, kmeans)


# === Train SVM classifier ===
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_hist, y_train_hist)

# === Predict on test set ===
y_pred = svm.predict(X_test_hist)

# === Evaluate ===
acc = accuracy_score(y_test_p, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test_p, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation='vertical')
plt.title("Confusion Matrix - ORB + BoVW + SVM")
plt.tight_layout()
plt.savefig("results/orb_svm_confusion_matrix.png")
plt.show()