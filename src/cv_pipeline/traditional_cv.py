#!/usr/bin/env python
"""
Run BoVW + SVM grid on iCubWorld IROS-16 subset.
Usage:
    python bow_grid.py --detector sift   # or orb
"""
import argparse, joblib, pathlib, cv2, numpy as np, tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
DATA_DIR = pathlib.Path('../data/iCubWorld')

def parse():
    ps = argparse.ArgumentParser()
    ps.add_argument('--detector', default='sift', choices=['sift','orb'])
    ps.add_argument('--ks', default='64,128,256,512')
    ps.add_argument('--Cs', default='0.1,1,10')
    return ps.parse_args()

def get_feats(det='sift'):
    cache = pathlib.Path(f'cache/{det}_desc.npy')
    if cache.exists(): return np.load(cache, allow_pickle=True).item()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if det=='sift': extractor=cv2.SIFT_create()
    else: extractor=cv2.ORB_create(nfeatures=2000)
    descs, labels = [], []
    for p in tqdm.tqdm(list((DATA_DIR/'train').rglob('*.jpg'))):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        kp, d  = extractor.detectAndCompute(img,None)
        if d is not None:
            descs.append(d); labels.append(int(p.parent.name))
    np.save(cache, {'descs':descs,'labels':labels})
    return {'descs':descs,'labels':labels}

def build_bow(descs, k):
    all_d = np.vstack(descs)
    km = MiniBatchKMeans(k, batch_size=20*k).fit(all_d)
    return km

def make_hist(km, descs):
    h = np.zeros((len(descs), km.n_clusters), dtype=np.float32)
    for i,d in enumerate(descs):
        if d is not None:
            idx = km.predict(d)
            for j in idx: h[i,j]+=1
    h = h / (h.sum(1,keepdims=True)+1e-7)
    return h

def main():
    args=parse(); ks=[int(x) for x in args.ks.split(',')]; Cs=[float(x) for x in args.Cs.split(',')]
    data = get_feats(args.detector)
    results=[]
    for k in ks:
        km = build_bow(data['descs'], k)
        H = make_hist(km, data['descs'])
        for C in Cs:
            clf = LinearSVC(C=C).fit(H, data['labels'])
            acc = accuracy_score(data['labels'], clf.predict(H))
            results.append((args.detector,k,C,acc))
            print(f'{args.detector} k={k} C={C}: {acc:.3f}')
            joblib.dump(clf, f'models/svm_{args.detector}_k{k}_C{C}.joblib')
    np.savetxt('results/bow_grid.csv', np.array(results, object), fmt='%s', delimiter=',')
if __name__=='__main__':
    main()
