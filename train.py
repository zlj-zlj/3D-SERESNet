# coding: utf-8
# #wo写的代码-患者独立性
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc, f1_score, confusion_matrix
import numpy as np
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from scipy.special import binom
from torch.utils.data import Dataset, DataLoader
import time
import os
import warnings
from datetime import datetime
from models.models import *
import argparse
from tqdm import tqdm
import json
from imblearn.metrics import sensitivity_score, specificity_score



warnings.filterwarnings("ignore")



def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch['data'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = sensitivity_score(all_labels, all_preds)
    specificity = specificity_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_score = calculate_auc(all_labels, all_probs)

    return avg_loss, accuracy, sensitivity, specificity, auc_score, f1


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs, labels = batch['data'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = sensitivity_score(all_labels, all_preds)
    specificity = specificity_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_score = calculate_auc(all_labels, all_probs)

    return avg_loss, accuracy, sensitivity, specificity, auc_score, f1




def main(args):

    BATCH_SIZE = 32
    EPOCHS = 100
    MODEL_NAME = "SEFRBnet1214"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data_base_dir = args.i

    patients = [d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d))]


    path_base = args.o


    for patient in patients:
        patient_dir = os.path.join(data_base_dir, patient)
        save_res_dir = os.path.join(path_base, patient)

        folds = [d for d in os.listdir(patient_dir) if
                 os.path.isdir(os.path.join(patient_dir, d)) and d.startswith('fold_')]
        inter_time = 0
        for fold in folds:

            fold_dir = os.path.join(patient_dir, fold)
            save_dir = os.path.join(save_res_dir, fold)
            results_epoch_file = os.path.join(save_dir, f'{patient}_{fold}_epoch_results.txt')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


            try:
                train_data = np.load(os.path.join(fold_dir, 'train_data.npy'))
                train_labels = np.load(os.path.join(fold_dir, 'train_labels.npy'))
                val_data = np.load(os.path.join(fold_dir, 'val_data.npy'))
                val_labels = np.load(os.path.join(fold_dir, 'val_labels.npy'))




                print(
                    f"  训练集 - 总样本: {len(train_labels)}, 前期: {np.sum(train_labels == 1)}, 间期: {np.sum(train_labels == 0)}")
                print(
                    f"  验证集 - 总样本: {len(val_labels)}, 前期: {np.sum(val_labels == 1)}, 间期: {np.sum(val_labels == 0)}")

            except FileNotFoundError as e:
                print(f"  文件缺失: {e}")
                continue


            train_dataset = EEGDataset(train_data, train_labels)
            val_dataset = EEGDataset(val_data, val_labels)


            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

            print("dataloader over")


            model = SEFRBNet().to(device)  # ResNet18


            train_counts = np.bincount(np.ravel(train_labels))
            w = train_counts[1] / (train_counts[0] + train_counts[1])
            print('w:', w)

            criterion = FocalLoss(alpha=w, gamma=2)
            optimizer = optim.AdamW(params=model.parameters(), lr=0.003, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,
                                                                                eta_min=0.003 / 10)




            best_val_acc = 0.0
            for epoch in range(EPOCHS):
                print(f"    Epoch {epoch + 1}/{EPOCHS}")


                train_loss, train_acc, train_sensitivity, train_specificity, train_auc, f1 = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                lr_scheduler.step()

                val_loss, val_acc, val_sensitivity, val_specificity, val_auc, f1 = validate_epoch(
                    model, val_loader, criterion, device
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_model_path_acc)



                if epoch == 0:
                    with open(results_epoch_file, 'w', encoding='utf-8') as f:
                        f.write(
                            "Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc,Train_Se,Val_Se,Train_Sp,Val_Sp,Train_AUC,Val_AUC\n")
                with open(results_epoch_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch + 1},"
                            f"{train_loss:.6f},{val_loss:.6f},"
                            f"{train_acc:.6f},{val_acc:.6f},"
                            f"{train_sensitivity:.6f},{val_sensitivity:.6f},"
                            f"{train_specificity:.6f},{val_specificity:.6f},"
                            f"{train_auc:.6f},{val_auc:.6f}\n")




if __name__ == '__main__':


        parser = argparse.ArgumentParser(description="Train SEFRBNet model.")
        parser.add_argument('--i', type=str, required=True, help="path of the data")
        parser.add_argument('--o', type=str, required=True, help="Path to the output directory.")
        args = parser.parse_args()

        main(args)
