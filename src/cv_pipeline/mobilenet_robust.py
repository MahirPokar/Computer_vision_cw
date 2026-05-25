#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

# Hyperparameter configurations
PARAMS = [
    {'backbone':'resnet18','batch_size':32,'lr':4.429696083150982e-05,'wd':1.875384922660738e-05,'optimizer':'AdamW','scheduler':'step','freeze':False},
    {'backbone':'resnet18','batch_size':64,'lr':0.00014405863525901197,'wd':1.1343938007490874e-05,'optimizer':'AdamW','scheduler':'step','freeze':False},
    {'backbone':'resnet34','batch_size':32,'lr':6.975851791909016e-05,'wd':3.7024310377601224e-06,'optimizer':'Adam','scheduler':'step','freeze':False},
    {'backbone':'resnet34','batch_size':64,'lr':3.44122616902724e-05,'wd':2.5944824935151936e-06,'optimizer':'Adam','scheduler':'cosine','freeze':False},
    {'backbone':'mobilenet_v2','batch_size':32,'lr':6.214780350640936e-05,'wd':0.0001387574962308744,'optimizer':'Adam','scheduler':'step','freeze':False},
    {'backbone':'mobilenet_v2','batch_size':64,'lr':0.004301588444411144,'wd':1.3367391542221841e-06,'optimizer':'SGD','scheduler':'step','freeze':False},
]

# Distortion functions

def occlude(img, occ_size=0.3):
    _,h,w = img.shape
    occ_h, occ_w = int(h*occ_size), int(w*occ_size)
    y = np.random.randint(0, h-occ_h)
    x = np.random.randint(0, w-occ_w)
    img[:, y:y+occ_h, x:x+occ_w] = 0
    return img

def add_noise(img, var=20):
    noise = torch.randn_like(img) * (var**0.5/255.0)
    return torch.clamp(img + noise, 0, 1)

def blur(img, k=7):
    np_img = img.permute(1,2,0).cpu().numpy()*255
    np_img = cv2.GaussianBlur(np_img,(((k//2)*2+1),)*2,0)
    return torch.from_numpy(np_img/255.0).permute(2,0,1).to(img.device)

# Dataset class
class ImageFolder(Dataset):
    def __init__(self, root, transform):
        if not os.path.isdir(root):
            print(f"Error: '{root}' not found.")
            sys.exit(1)
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        if not classes:
            print(f"Error: no classes in '{root}'.")
            sys.exit(1)
        self.paths, self.labels = [], []
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            for f in os.listdir(os.path.join(root,c)):
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.ppm')):
                    self.paths.append(os.path.join(root,c,f))
                    self.labels.append(self.class_to_idx[c])
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

# Train/test routines

def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); running = 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad(); out = model(imgs)
        loss = criterion(out, labs); loss.backward(); optimizer.step()
        running += loss.item()*imgs.size(0)
    return running/len(loader.dataset)

def test_epoch(model, loader, device, distort=None):
    model.eval(); correct = 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            if distort == 'occlusion': imgs = torch.stack([occlude(i.clone()) for i in imgs])
            elif distort == 'noise': imgs = torch.stack([add_noise(i.clone()) for i in imgs])
            elif distort == 'blur': imgs = torch.stack([blur(i.clone()) for i in imgs])
            out = model(imgs); preds = out.argmax(1)
            correct += (preds.cpu() == labs).sum().item()
    return correct/len(loader.dataset)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = ImageFolder(args.train_dir, transform)
    test_ds  = ImageFolder(args.test_dir, transform)

    results_all = {}
    for cfg in PARAMS:
        name = f"{cfg['backbone']}_bs{cfg['batch_size']}_opt{cfg['optimizer']}"
        print(f"\n=== Training {name} ===")
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'])

        # Build model
        if cfg['backbone']=='resnet18': model = models.resnet18(pretrained=False)
        elif cfg['backbone']=='resnet34': model = models.resnet34(pretrained=False)
        else: model = models.mobilenet_v2(pretrained=False)
        # adjust classifier
        if 'resnet' in cfg['backbone']:
            in_f = model.fc.in_features; model.fc = nn.Linear(in_f,len(train_ds.class_to_idx))
        else:
            in_f = model.classifier[-1].in_features; model.classifier[-1] = nn.Linear(in_f,len(train_ds.class_to_idx))
        model = model.to(device)
        if cfg['freeze']:
            for param in model.parameters(): param.requires_grad = False
            head = model.fc if 'resnet' in cfg['backbone'] else model.classifier[-1]
            for param in head.parameters(): param.requires_grad = True

        # Optimizer
        if cfg['optimizer']=='AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        elif cfg['optimizer']=='Adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'], momentum=0.9)
        # Scheduler
        if cfg['scheduler']=='step':
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs+1):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            print(f"Epoch {epoch}/{args.epochs}, Loss={loss:.4f}")

        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f"models/{name}.pth")

        res = {}
        for dist in ['original','occlusion','noise','blur']:
            acc = test_epoch(model, test_loader, device, distort=(None if dist=='original' else dist))
            res[dist] = acc * 100
            print(f"{dist}: {res[dist]:.2f}%")
        results_all[name] = res

    np.save('models/cnn_robustness.npy', results_all, allow_pickle=True)
    print("Saved CNN robustness results to models/cnn_robustness.npy")