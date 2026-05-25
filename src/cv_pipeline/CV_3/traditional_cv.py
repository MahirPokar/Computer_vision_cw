import os
import cv2
import numpy as np
import joblib
import json
import csv
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Directories for caching and outputs
dirs = {
    'cache': './cache',
    'models': './models',
    'results': './results'
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Parameter grids
descriptor_types = ['ORB', 'SIFT']
k_clusters_list = [50, 100, 200]
svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Utility functions for caching
def cache_path(name, ext, directory):
    return os.path.join(dirs[directory], f"{name}.{ext}")

# 1. Load images and labels (recursive) 
def load_images_and_labels(base_path):
    image_paths, labels = [], []
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(base_path, cls)
        # walk all nested folders
        for root, _, files in os.walk(cls_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                    image_paths.append(os.path.join(root, fname))
                    labels.append(idx)
    return image_paths, labels, class_names

# 2. Extract descriptors (with caching)
def extract_descriptors(image_paths, descriptor_type):
    cache_file = cache_path(f"descriptors_{descriptor_type}", 'pkl', 'cache')
    if os.path.exists(cache_file):
        print(f"Loading cached descriptors for {descriptor_type}...")
        all_desc, desc_per_image = joblib.load(cache_file)
        return all_desc, desc_per_image

    print(f"Extracting {descriptor_type} descriptors...")
    if descriptor_type == 'SIFT':
        descriptor = cv2.SIFT_create()
    else:
        descriptor = cv2.ORB_create(nfeatures=500)

    all_desc = []
    desc_per_image = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            desc_per_image.append(None)
            continue
        _, desc = descriptor.detectAndCompute(img, None)
        desc_per_image.append(desc)
        if desc is not None:
            all_desc.extend(desc)
    all_desc = np.array(all_desc)
    joblib.dump((all_desc, desc_per_image), cache_file)
    return all_desc, desc_per_image

# 3. Build or load KMeans vocabulary
def get_kmeans(descriptors, descriptor_type, k):
    name = f"kmeans_{descriptor_type}_{k}"
    cache_file = cache_path(name, 'joblib', 'models')
    if os.path.exists(cache_file):
        print(f"Loading cached KMeans for {descriptor_type}, k={k}...")
        return joblib.load(cache_file)

    print(f"Fitting KMeans for {descriptor_type}, k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, verbose=0)
    kmeans.fit(descriptors)
    joblib.dump(kmeans, cache_file)
    return kmeans

# 4. Compute histograms of visual words
def compute_histograms(kmeans, desc_per_image, k):
    histograms = []
    for desc in desc_per_image:
        if desc is None:
            hist = np.zeros(k)
        else:
            words = kmeans.predict(desc)
            hist, _ = np.histogram(words, bins=np.arange(k+1))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
        histograms.append(hist)
    return np.array(histograms)

# 5. Train and evaluate, saving results
def train_and_evaluate(X, y, descriptor_type, k):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    svc = SVC()
    grid = GridSearchCV(svc, svm_param_grid, cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    params = grid.best_params_
    y_pred = best.predict(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save model and metrics
    model_path = cache_path(f"svm_{descriptor_type}_{k}", 'joblib', 'models')
    joblib.dump(best, model_path)
    with open(cache_path(f"report_{descriptor_type}_{k}", 'json', 'results'), 'w') as f:
        json.dump(report, f, indent=2)
    np.save(cache_path(f"cm_{descriptor_type}_{k}", 'npy', 'results'), cm)

    # Append summary to CSV
    csv_file = cache_path('results_summary', 'csv', 'results')
    header = ['descriptor', 'k', 'C', 'kernel', 'accuracy']
    row = [descriptor_type, k, params['C'], params['kernel'], report['accuracy']]
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# Main routine
def main():
    base_path = '/home/mscrobotics2425laptop24/Documents/SEM_2/comp_vision_2/CV_3/dataset/iCubWorld/train'
    image_paths, labels, class_names = load_images_and_labels(base_path)
    print(f"Loaded {len(image_paths)} images across {len(class_names)} classes.")

    # Loop over descriptor types
    for descriptor in descriptor_types:
        all_desc, desc_per_image = extract_descriptors(image_paths, descriptor)
        print(f"Total descriptors for {descriptor}: {all_desc.shape[0]}")

        # Loop over cluster sizes
        for k in k_clusters_list:
            try:
                kmeans = get_kmeans(all_desc, descriptor, k)
                hists = compute_histograms(kmeans, desc_per_image, k)
                train_and_evaluate(hists, labels, descriptor, k)
                print(f"Completed run: {descriptor}, k={k}")
            except Exception as e:
                # Log errors
                err_file = cache_path('error_log', 'txt', 'results')
                with open(err_file, 'a') as ef:
                    ef.write(f"Error with {descriptor}, k={k}: {e}\n")
                print(f"Error on {descriptor}, k={k}. Check {err_file}.")

if __name__ == '__main__':
    main()
