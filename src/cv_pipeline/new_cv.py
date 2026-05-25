#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
import argparse
import random
from tqdm import tqdm
import sys

# Constants
CLUSTERS = 500
DETECTORS = ['sift', 'orb']
DISTORTIONS = {
    'original': lambda img: img,
    'occlusion': lambda img, s=0.3: cv2.rectangle(
        img.copy(),
        (int(img.shape[1]*0.35), int(img.shape[0]*0.35)),
        (int(img.shape[1]*0.65), int(img.shape[0]*0.65)),
        (0,0,0), -1
    ),
    'noise': lambda img, v=20: cv2.add(
        img,
        np.random.normal(0, v**0.5, img.shape).astype('uint8')
    ),
    'blur': lambda img, k=7: cv2.GaussianBlur(
        img,
        (((k//2)*2+1), ((k//2)*2+1)),
        0
    )
}

# Extract descriptors for a list of image paths

def extract_descriptors(image_paths, detector_name):
    if detector_name == 'sift':
        det = cv2.SIFT_create()
    else:
        det = cv2.ORB_create(nfeatures=2000)
    all_descs = []
    img_descs = []
    for p in tqdm(image_paths, desc=f"Extract {detector_name}"):
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, d = det.detectAndCompute(gray, None)
        if d is None:
            img_descs.append(np.zeros((0, det.descriptorSize()), dtype=np.float32))
        else:
            img_descs.append(d)
            all_descs.append(d)
    if len(all_descs) > 0:
        return img_descs, np.vstack(all_descs)
    else:
        return img_descs, np.empty((0, det.descriptorSize()), dtype=np.float32)

# Build normalized histograms from descriptors

def build_histograms(img_descs, kmeans):
    n_clusters = kmeans.n_clusters
    hists = np.zeros((len(img_descs), n_clusters), dtype=np.float32)
    for i, d in enumerate(img_descs):
        if d.shape[0] > 0:
            idx = kmeans.predict(d)
            for j in idx:
                hists[i, j] += 1
            hists[i] /= hists[i].sum()
    return hists

# Load image file paths and integer labels from directory

def load_dataset_paths(data_dir):
    data_dir = Path(data_dir)
    classes = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        class_dir = data_dir / cls
        for imgf in class_dir.glob('*.*'):
            paths.append(imgf)
            labels.append(idx)
    return paths, np.array(labels), classes

# Main pipeline

def main(args):
    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load train/test file lists
    train_paths, train_labels, classes = load_dataset_paths(args.train_dir)
    test_paths, test_labels, _ = load_dataset_paths(args.test_dir)
    print(f"Classes: {classes}")
    print(f"Train images: {len(train_paths)}, Test images: {len(test_paths)}")

    # Validate paths
    if len(train_paths) == 0:
        print(f"Error: no training images found in '{args.train_dir}'.")
        sys.exit(1)
    if len(test_paths) == 0:
        print(f"Error: no test images found in '{args.test_dir}'.")
        sys.exit(1)

    results = {}

    for detector_name in DETECTORS:
        print(f"\n== {detector_name.upper()} Pipeline ==")

        # Extract and cluster descriptors
        train_descs_list, all_train_descs = extract_descriptors(train_paths, detector_name)
        kmeans = MiniBatchKMeans(
            n_clusters=CLUSTERS,
            batch_size=CLUSTERS * 10,
            random_state=42
        )
        kmeans.fit(all_train_descs)
        joblib.dump(
            kmeans,
            args.output_dir / f"kmeans_{detector_name}.joblib"
        )
        print(f"Saved KMeans (k={CLUSTERS}) for {detector_name}")

        # Train SVM on BoVW histograms
        train_hists = build_histograms(train_descs_list, kmeans)
        clf = LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=42
        )
        clf.fit(train_hists, train_labels)
        joblib.dump(
            clf,
            args.output_dir / f"svm_{detector_name}.joblib"
        )
        print(f"Saved SVM for {detector_name}")

        # Test on distortions
        det = cv2.SIFT_create() if detector_name == 'sift' else cv2.ORB_create(nfeatures=2000)
        accuracies = {d: [] for d in DISTORTIONS}
        for img_path, gt in tqdm(
            zip(test_paths, test_labels),
            total=len(test_paths),
            desc=f"Testing {detector_name}"
        ):
            orig = cv2.imread(str(img_path))
            for dname, func in DISTORTIONS.items():
                img = func(orig.copy()) if dname != 'original' else orig
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, d = det.detectAndCompute(gray, None)
                hist = np.zeros(CLUSTERS, dtype=np.float32)
                if d is not None and d.shape[0] > 0:
                    idx = kmeans.predict(d)
                    for j in idx:
                        hist[j] += 1
                    hist /= hist.sum()
                pred = clf.predict(hist.reshape(1, -1))[0]
                accuracies[dname].append(pred == gt)

        # Compute and print accuracies
        det_results = {d: float(np.mean(accuracies[d])) for d in DISTORTIONS}
        results[detector_name] = det_results
        print(f"Results {detector_name}: {det_results}")

    # Save overall results
    np.save(
        args.output_dir / "results.npy",
        results,
        allow_pickle=True
    )
    print(f"Saved aggregate results to {args.output_dir / 'results.npy'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and evaluate BoVW pipelines with SIFT and ORB"
    )
    parser.add_argument(
        '--train_dir',
        default=r"C:\Users\Mahir\Documents\COMPVISION\datatset\iCubWorld\test\categorization",
        help="Training data directory with class subfolders"
    )
    parser.add_argument(
        '--test_dir',
        default=r"C:\Users\Mahir\Documents\COMPVISION\datatset\iCubWorld\test\categorization",
        help="Test data directory with class subfolders"
    )
    parser.add_argument(
        '--output_dir',
        default=Path(r"C:\Users\Mahir\Documents\COMPVISION\models"),
        type=Path,
        help="Directory to save models and results"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
