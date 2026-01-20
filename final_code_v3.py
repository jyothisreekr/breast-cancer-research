#!/usr/bin/env python3
"""
Automated Breast Cancer Diagnosis Pipeline
- Preprocessing: Resize, CLAHE, Normalization
- IGAC-like segmentation (approximation using skimage morphological chan-vese/threshold + morphology)
- LIPT feature extraction: statistical features, LBP, GLCM
- ResNet18 classifier with appended handcrafted features
- Training / Validation / Testing
- Metrics: confusion matrix, accuracy, sensitivity, specificity, precision, F1, TP/TN/FP/FN
- Segmentation metrics: Dice (DSC), IoU, Hausdorff Distance (HD), VOE, Mean Surface Distance (MSD)

Adjust dataset_root and other parameters below as needed.
"""

import os
import random
import shutil
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# skimage / scipy / sklearn
from skimage import exposure, filters, morphology, measure, segmentation, feature
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import morphological_chan_vese
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.metrics import hausdorff_distance
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Configuration
# ---------------------------
dataset_root = "/content/drive/MyDrive/BC_images"
images_dir = os.path.join(dataset_root, "images")
masks_dir = os.path.join(dataset_root, "masks")
labels_csv = os.path.join(dataset_root, "labels.csv")

img_size = (224, 224)
batch_size = 8
num_epochs = 25
learning_rate = 1e-4
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42
feature_dim = 64
num_classes = 2

np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

# ---------------------------
# Utility functions & Metrics
# ---------------------------

def read_image_grayscale(path):
    img = Image.open(path)
    img = img.convert("L")  # ensure grayscale
    arr = np.array(img)
    return arr

def clahe_enhance(img_gray, clip_limit=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)

def normalize_image(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def igac_segment(img_gray, resize_to=img_size, sigma=1.0):
    """
    Approximate IGAC:
      - CLAHE
      - Gaussian blur
      - Otsu threshold
      - Morphological closing/opening
      - (Optional) morphological Chan-Vese refinement
    Returns binary mask (0/1)
    """
    img_clahe = clahe_enhance(img_gray)
    img_blur = cv2.GaussianBlur(img_clahe, (0,0), sigmaX=sigma)
    # Otsu threshold
    try:
        thresh = threshold_otsu(img_blur)
    except Exception:
        thresh = np.mean(img_blur)
    mask = (img_blur >= thresh).astype(np.uint8)
    # Keep largest connected component(s)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=100)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.binary_opening(mask, morphology.disk(3))
    # refine with morphological chan-vese (fast) for some iterations
    try:
        mc = morphological_chan_vese(img_blur/255.0, iterations=30, init_level_set=mask.astype(np.int8))
        mask = mc.astype(bool)
    except Exception:
        mask = mask
    mask = mask.astype(np.uint8)
    return mask

def dice_coef(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() + gt.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum())

def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return inter / union

def voe_score(pred, gt):
    return 1.0 - iou_score(pred, gt)

def hausdorff(pred, gt):
    # skimage hausdorff_distance expects binary images with same shape
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    try:
        return hausdorff_distance(pred.astype(bool), gt.astype(bool))
    except Exception:
        return np.nan

def mean_surface_distance(pred, gt):
    # compute mean distance between surface points of pred and gt
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    # distance transform from boundaries
    pred_border = segmentation.find_boundaries(pred, mode='outer')
    gt_border = segmentation.find_boundaries(gt, mode='outer')
    pred_pts = np.column_stack(np.nonzero(pred_border))
    gt_pts = np.column_stack(np.nonzero(gt_border))
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    d1 = cdist(pred_pts, gt_pts).min(axis=1)
    d2 = cdist(gt_pts, pred_pts).min(axis=1)
    return np.mean(np.concatenate([d1, d2]))

# ---------------------------
# Feature extraction (LIPT-style)
# ---------------------------
def extract_lipt_features(img_gray, mask=None):
    """
    Return a 1D feature vector descriptive of the ROI.
    - statistical features: mean, std, skewness, kurtosis, percentiles
    - LBP histogram
    - GLCM properties (contrast, correlation, energy, homogeneity)
    """
    if mask is None:
        roi = img_gray
    else:
        roi = img_gray.copy()
        roi[mask == 0] = 0

    roi_f = roi.astype(np.float32)
    # basic stats
    mean = roi_f.mean()
    std = roi_f.std()
    mn = roi_f.min()
    mx = roi_f.max()
    p25 = np.percentile(roi_f, 25)
    p50 = np.percentile(roi_f, 50)
    p75 = np.percentile(roi_f, 75)
    # LBP
    lbp = local_binary_pattern(roi_f, P=8, R=1, method="uniform")
    # histogram of LBP (normalize)
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    # GLCM (quantize to 8 levels)
    try:
        img_q = (roi_f - roi_f.min()) / (roi_f.max() - roi_f.min() + 1e-8)
        img_q = (img_q * 7).astype(np.uint8)
        glcm = greycomatrix(img_q, [1], [0], levels=8, symmetric=True, normed=True)
        contrast = greycoprops(glcm, "contrast")[0,0]
        correlation = greycoprops(glcm, "correlation")[0,0]
        energy = greycoprops(glcm, "energy")[0,0]
        homogeneity = greycoprops(glcm, "homogeneity")[0,0]
    except Exception:
        contrast = correlation = energy = homogeneity = 0.0

    feats = np.concatenate([[mean, std, mn, mx, p25, p50, p75],
                            lbp_hist,
                            [contrast, correlation, energy, homogeneity]])
    # if mask provided, add mask area ratio
    if mask is not None:
        feats = np.concatenate([feats, [mask.sum() / (mask.size + 1e-8)]])
    return feats.astype(np.float32)

# ---------------------------
# Dataset
# ---------------------------
class MammogramDataset(Dataset):
    def __init__(self, df, images_dir, masks_dir=None, transform=None, preprocess=True):
        """
        df: dataframe with columns filename,label
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        label = int(row['label'])
        img_path = os.path.join(self.images_dir, fname)
        img = read_image_grayscale(img_path)
        # Preprocess: resize, CLAHE, normalize
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
        img_clahe = clahe_enhance(img_resized)
        img_norm = normalize_image(img_clahe)
        # segmentation (IGAC approximation)
        mask = igac_segment((img_resized).astype(np.uint8))
        mask_resized = cv2.resize((mask*255).astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST) // 255
        # extract features
        feats = extract_lipt_features((img_resized).astype(np.uint8), mask_resized)
        # transform to tensor for ResNet input (3 channels)
        # replicate grayscale to 3 channels
        img_3c = np.stack([img_norm, img_norm, img_norm], axis=-1)  # HWC
        if self.transform:
            img_t = self.transform(Image.fromarray((img_3c*255).astype(np.uint8)))
        else:
            # default to tensor conversion and normalization
            t = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            img_t = t(Image.fromarray((img_3c*255).astype(np.uint8)))
        return img_t, torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long), mask_resized, fname

# ---------------------------
# Model: ResNet with appended features
# ---------------------------
class ResNetWithFeatures(nn.Module):
    def __init__(self, feat_dim_in, feat_embed_dim=feature_dim, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        # remove fc
        self.backbone.fc = nn.Identity()
        # embed features
        self.feat_embed = nn.Sequential(
            nn.Linear(feat_dim_in, feat_embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features + feat_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_feat):
        x_img = self.backbone(x_img)  # batch x in_features
        x_feat = self.feat_embed(x_feat)
        x = torch.cat([x_img, x_feat], dim=1)
        out = self.classifier(x)
        return out

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, feats, labels, _, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(labels.detach().cpu().numpy()))
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, feats, labels, _, _ in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(imgs, feats)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(labels.detach().cpu().numpy()))
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1, cm, all_labels, all_preds

# ---------------------------
# Main run
# ---------------------------
def main():
    # Load labels CSV
    df = pd.read_csv(labels_csv)
    # ensure columns
    assert 'filename' in df.columns and 'label' in df.columns, "labels.csv must have columns filename,label"
    # shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # splits: 70/15/15
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    df_test = df.iloc[n_train+n_val:].reset_index(drop=True)

    print(f"Samples: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # determine feature dimension by extracting one sample
    sample_img_path = os.path.join(images_dir, df_train.iloc[0]['filename'])
    sample_img = read_image_grayscale(sample_img_path)
    sample_mask = igac_segment(sample_img)
    sample_feats = extract_lipt_features(cv2.resize(sample_img, img_size), cv2.resize(sample_mask.astype(np.uint8), img_size))
    feat_dim = len(sample_feats)
    print("Feature vector length:", feat_dim)

    # transforms
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_ds = MammogramDataset(df_train, images_dir, masks_dir if os.path.exists(masks_dir) else None, transform=transform)
    val_ds = MammogramDataset(df_val, images_dir, masks_dir if os.path.exists(masks_dir) else None, transform=transform)
    test_ds = MammogramDataset(df_test, images_dir, masks_dir if os.path.exists(masks_dir) else None, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ResNetWithFeatures(feat_dim, feat_embed_dim=feature_dim, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_prec':[], 'val_rec':[], 'val_f1':[]}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm, _, _ = eval_model(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        history['val_f1'].append(val_f1)

        print(f" Train loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f" Val   loss={val_loss:.4f} acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f}")
        print(" Val Confusion Matrix:\n", val_cm)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(" Saved best model.")

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend(); plt.title("Accuracy")
    plt.savefig("training_curves.png", bbox_inches='tight')
    plt.close()

    # Load best model for test
    model.load_state_dict(torch.load("best_model.pth"))
    _, test_acc, test_prec, test_rec, test_f1, test_cm, test_labels, test_preds = eval_model(model, test_loader, criterion)
    print("\nTest Results:")
    print(f" Accuracy: {test_acc:.4f}")
    print(f" Precision: {test_prec:.4f}")
    print(f" Recall/Sensitivity: {test_rec:.4f}")
    # Specificity: TN / (TN + FP)
    tn, fp, fn, tp = test_cm.ravel() if test_cm.size == 4 else (0,0,0,0)
    specificity = tn / (tn + fp + 1e-8) if (tn + fp) > 0 else 0.0
    print(f" Specificity: {specificity:.4f}")
    print(f" F1: {test_f1:.4f}")
    print("Confusion Matrix:\n", test_cm)
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Save confusion matrix plot
    plt.figure(figsize=(4,4))
    plt.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, ["benign","malignant"])
    plt.yticks(tick_marks, ["benign","malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            plt.text(j, i, str(test_cm[i,j]), ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Segmentation metrics on test set if masks available
    segmentation_metrics = []
    if os.path.exists(masks_dir):
        print("Computing segmentation metrics on test set (IGAC vs. ground truth masks)...")
        for imgs, feats, labels, masks_pred, fnames in tqdm(test_loader):
            # masks_pred are batch of predicted igac masks from dataset (we computed earlier)
            # load ground truth masks
            for i, fname in enumerate(fnames):
                gt_mask_path = os.path.join(masks_dir, fname)
                if not os.path.exists(gt_mask_path):
                    continue
                gt_mask = read_image_grayscale(gt_mask_path)
                gt_mask = cv2.resize((gt_mask>0).astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST)
                pred_mask = masks_pred[i].numpy().astype(np.uint8)
                dsc = dice_coef(pred_mask, gt_mask)
                iou = iou_score(pred_mask, gt_mask)
                hd = hausdorff(pred_mask, gt_mask)
                voe = voe_score(pred_mask, gt_mask)
                msd = mean_surface_distance(pred_mask, gt_mask)
                segmentation_metrics.append({'filename': fname, 'dice': dsc, 'iou': iou, 'hd': hd, 'voe': voe, 'msd': msd})
        if len(segmentation_metrics) > 0:
            sm = pd.DataFrame(segmentation_metrics)
            print("Segmentation metrics summary:")
            print(sm.describe().loc[['mean','std','min','max']])
            sm.to_csv("segmentation_metrics.csv", index=False)
        else:
            print("No ground-truth masks found for test set.")

    # Save test results
    pd.DataFrame({'filename': df_test['filename'], 'true': test_labels, 'pred': test_preds}).to_csv("test_results.csv", index=False)
    print("Done. Artifacts saved: best_model.pth, training_curves.png, confusion_matrix.png, test_results.csv")

if __name__ == "__main__":
    main()
