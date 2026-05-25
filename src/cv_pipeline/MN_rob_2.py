#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

# Distortion function for occlusion
def occlude_tensor(img, occ_size=0.3):
    # img: Tensor CxHxW
    c, h, w = img.shape
    occ_h, occ_w = int(h * occ_size), int(w * occ_size)
    y = np.random.randint(0, h - occ_h)
    x = np.random.randint(0, w - occ_w)
    img[:, y:y+occ_h, x:x+occ_w] = 0
    return img

# Dataset that treats immediate subdirectories of root as classes and recursively loads images
class ImageFolder(Dataset):
    def __init__(self, root, transform, augment=False, occ_size=0.3):
        if not os.path.isdir(root):
            print(f"Error: '{root}' not found.")
            sys.exit(1)
        
        # Supported image extensions
        image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm')
        
        # First-level subdirectories under root are class names
        classes = [d for d in sorted(os.listdir(root)) 
                   if os.path.isdir(os.path.join(root, d))]
        if not classes:
            print(f"Error: no class subdirectories found in '{root}'.")
            sys.exit(1)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        # Gather all image file paths and labels by walking each class directory
        self.paths = []
        self.labels = []
        for cls_name in classes:
            class_dir = os.path.join(root, cls_name)
            for dirpath, _, filenames in os.walk(class_dir):
                for fname in filenames:
                    if fname.lower().endswith(image_exts):
                        fpath = os.path.join(dirpath, fname)
                        if os.path.isfile(fpath):
                            self.paths.append(fpath)
                            self.labels.append(self.class_to_idx[cls_name])
        
        if len(self.paths) == 0:
            print(f"Error: no images found in '{root}'.")
            sys.exit(1)
        
        print(f"Loaded {len(self.paths)} images from {len(classes)} classes in '{root}'")
        self.transform = transform
        self.augment = augment
        self.occ_size = occ_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        tensor = self.transform(img)
        if self.augment:
            tensor = occlude_tensor(tensor.clone(), self.occ_size)
        return tensor, self.labels[idx]

# Training and evaluation loops
def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); total_loss = 0.0
    for imgs, labs in tqdm(loader, desc='Training'):
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labs)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, augment=False, occ_size=0.3):
    model.eval(); correct = 0
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc='Evaluating'):
            imgs = imgs.to(device)
            if augment:
                imgs = torch.stack([occlude_tensor(i.clone(), occ_size) for i in imgs])
            out = model(imgs); preds = out.argmax(1)
            correct += (preds.cpu() == labs).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train/test CNN on categorization with occlusion augmentation')
    parser.add_argument(
        '--train_dir',
        default=r"C:\Users\Mahir\Documents\COMPVISION\datatset\iCubWorld\train",
        help='Train directory (root of class folders)'
    )
    parser.add_argument(
        '--test_dir',
        default=r"C:\Users\Mahir\Documents\COMPVISION\datatset\iCubWorld\test\categorization",
        help='Test directory (root of class folders)'
    )
    parser.add_argument('--backbone', choices=['mobilenet_v2','resnet18','resnet34'], default='mobilenet_v2')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--occ_size', type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Common transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets and loaders
    train_ds = ImageFolder(args.train_dir, transform, augment=True, occ_size=args.occ_size)
    test_ds_clean = ImageFolder(args.test_dir, transform, augment=False)
    test_ds_occ   = ImageFolder(args.test_dir, transform, augment=True, occ_size=args.occ_size)

    train_loader    = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader_clean = DataLoader(test_ds_clean, batch_size=args.batch_size)
    test_loader_occ   = DataLoader(test_ds_occ, batch_size=args.batch_size)

    # Build model
    if args.backbone == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, len(train_ds.class_to_idx))
    elif args.backbone == 'resnet18':
        model = models.resnet18(pretrained=False)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(train_ds.class_to_idx))
    else:
        model = models.resnet34(pretrained=False)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(train_ds.class_to_idx))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}")

    # Save
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{args.backbone}_aug.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Evaluate
    acc_clean = evaluate(model, test_loader_clean, device, augment=False)
    acc_occ   = evaluate(model, test_loader_occ, device, augment=False)
    print(f"Test Acc (clean):    {acc_clean*100:.2f}%")
    print(f"Test Acc (occluded): {acc_occ*100:.2f}%")
