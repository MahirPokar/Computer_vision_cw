#!/usr/bin/env python3
import os
import sys
import random
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Distortion function for occlusion
def occlude_tensor(img, occ_size=0.3):
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
        image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.ppm')
        classes = [d for d in sorted(os.listdir(root)) 
                   if os.path.isdir(os.path.join(root, d))]
        if not classes:
            print(f"Error: no class subdirectories found in '{root}'.")
            sys.exit(1)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.paths = []
        self.labels = []
        for cls_name in classes:
            for dirpath, _, filenames in os.walk(os.path.join(root, cls_name)):
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


def evaluate(model, loader, device, augment=False, occ_size=0.3, return_preds=False):
    model.eval(); correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc='Evaluating'):
            imgs = imgs.to(device)
            if augment:
                imgs = torch.stack([occlude_tensor(i.clone(), occ_size) for i in imgs])
            out = model(imgs); preds = out.argmax(1)
            correct += (preds.cpu() == labs).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labs.tolist())
    acc = correct / len(loader.dataset)
    if return_preds:
        return acc, all_labels, all_preds
    return acc


def plot_loss_acc(train_losses, acc_clean, acc_occ, out_dir):
    epochs = list(range(1, len(train_losses)+1))
    plt.figure()
    plt.plot(epochs, train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.savefig(os.path.join(out_dir, 'train_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, [a*100 for a in acc_clean], label='Clean')
    plt.plot(epochs, [a*100 for a in acc_occ], label='Occluded')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'test_accuracy.png'))
    plt.close()


def plot_confusion(cm, classes, title, out_path):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.savefig(out_path)
    plt.close()


def save_comparison_images(test_ds, out_dir, num=5, occ_size=0.3):
    os.makedirs(out_dir, exist_ok=True)
    idxs = random.sample(range(len(test_ds)), num)
    for i, idx in enumerate(idxs):
        img_path = test_ds.paths[idx]
        img = Image.open(img_path).convert('RGB')
        clean = test_ds.transform(img)
        occluded = occlude_tensor(clean.clone(), occ_size)
        # convert tensors back to PIL for saving
        clean_pil = transforms.ToPILImage()(clean)
        occ_pil = transforms.ToPILImage()(occluded)
        # side by side
        combined = Image.new('RGB', (clean_pil.width*2, clean_pil.height))
        combined.paste(clean_pil, (0,0)); combined.paste(occ_pil, (clean_pil.width,0))
        combined.save(os.path.join(out_dir, f'comparison_{i}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train/test CNN and generate plots & comparison images')
    parser.add_argument('--train_dir', default='datatset\iCubWorld\\train', help='Train directory')
    parser.add_argument('--test_dir', default='datatset\iCubWorld\\test\categorization', help='Test directory')
    parser.add_argument('--backbone', choices=['mobilenet_v2','resnet18','resnet34'], default='mobilenet_v2')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--occ_size', type=float, default=0.3)
    parser.add_argument('--out_dir', default='results', help='Output directory for figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = ImageFolder(args.train_dir, transform, augment=True, occ_size=args.occ_size)
    test_ds_clean = ImageFolder(args.test_dir, transform, augment=False)
    test_ds_occ = ImageFolder(args.test_dir, transform, augment=True, occ_size=args.occ_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader_clean = DataLoader(test_ds_clean, batch_size=args.batch_size)
    test_loader_occ = DataLoader(test_ds_occ, batch_size=args.batch_size)

    # build model
    if args.backbone == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        in_f = model.classifier[-1].in_features; model.classifier[-1] = nn.Linear(in_f, len(train_ds.class_to_idx))
    elif args.backbone == 'resnet18':
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features; model.fc = nn.Linear(in_f, len(train_ds.class_to_idx))
    else:
        model = models.resnet34(weights=None)
        in_f = model.fc.in_features; model.fc = nn.Linear(in_f, len(train_ds.class_to_idx))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # training + evaluation
    train_losses, acc_clean_list, acc_occ_list = [], [], []
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        scheduler.step()
        acc_clean = evaluate(model, test_loader_clean, device, augment=False)
        acc_occ = evaluate(model, test_loader_occ, device, augment=False)
        acc_clean_list.append(acc_clean)
        acc_occ_list.append(acc_occ)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} | Clean Acc: {acc_clean*100:.2f}% | Occ Acc: {acc_occ*100:.2f}%")

    # save model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f"{args.backbone}_aug.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # generate plots
    plot_loss_acc(train_losses, acc_clean_list, acc_occ_list, args.out_dir)

    # confusion matrices
    acc_c, labels_c, preds_c = evaluate(model, test_loader_clean, device, augment=False, return_preds=True)
    cm_c = confusion_matrix(labels_c, preds_c)
    plot_confusion(cm_c, list(train_ds.class_to_idx.keys()), 'Confusion Matrix (Clean)', os.path.join(args.out_dir,'cm_clean.png'))

    acc_o, labels_o, preds_o = evaluate(model, test_loader_occ, device, augment=False, return_preds=True)
    cm_o = confusion_matrix(labels_o, preds_o)
    plot_confusion(cm_o, list(train_ds.class_to_idx.keys()), 'Confusion Matrix (Occluded)', os.path.join(args.out_dir,'cm_occluded.png'))

    # example comparisons
    save_comparison_images(test_ds_clean, os.path.join(args.out_dir,'comparisons'), num=5, occ_size=args.occ_size)

    print(f"Figures and comparison images saved in '{args.out_dir}/'")
