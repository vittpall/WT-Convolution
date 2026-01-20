#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

from resnet_WTHaar import ResNet20

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------
# HuggingFace CIFAR-10 Dataset
# ------------------------------------------------------------------
class HFCIFAR10(Dataset):
    def __init__(self, split, transform=None):
        self.ds = load_dataset("cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["img"]
        label = self.ds[idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar10_loaders(batch_size, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    train_ds = HFCIFAR10("train", transform_train)
    test_ds = HFCIFAR10("test", transform_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=len(dataloader))
    count = 0
    train_loss = 0.0
    train_acc = 0.0

    for x, labels in pbar:
        x, labels = x.to(device), labels.to(device)

        outputs = model(x)
        loss = loss_fn(outputs, labels)

        _, pred = torch.max(outputs, 1)
        num_correct = (pred == labels).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        acc_val = num_correct.item() / len(labels)

        count += len(labels)
        train_loss += loss_val * len(labels)
        train_acc += num_correct.item()

        pbar.set_description(
            f"loss: {loss_val:.4f}, acc: {acc_val:.4f}, [{count}/{size}]"
        )

    return train_loss / count, train_acc / count

# ------------------------------------------------------------------
# Test Loop
# ------------------------------------------------------------------
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=len(dataloader))
    count = 0
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for x, labels in pbar:
            x, labels = x.to(device), labels.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, labels)

            _, pred = torch.max(outputs, 1)
            num_correct = (pred == labels).sum()

            loss_val = loss.item()
            acc_val = num_correct.item() / len(labels)

            count += len(labels)
            test_loss += loss_val * len(labels)
            test_acc += num_correct.item()

            pbar.set_description(
                f"loss: {loss_val:.4f}, acc: {acc_val:.4f}, [{count}/{size}]"
            )

    return test_loss / count, test_acc / count

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    save_dir = "./saved"
    batchsize = 128
    n_epochs = 200
    lr = 0.1
    evaluate_train = False

    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_loader, test_loader = get_cifar10_loaders(batchsize)

    # Model
    model = ResNet20().to(device)
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer (same logic as original)
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if "bias" in name:
            bias_p.append(p)
        else:
            weight_p.append(p)

    optimizer = torch.optim.SGD(
        [
            {"params": weight_p, "weight_decay": 1e-4},
            {"params": bias_p, "weight_decay": 0},
        ],
        lr=lr,
        momentum=0.9,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[82, 122, 163], gamma=0.1
    )

    # Logs
    log_train_loss, log_train_acc = [], []
    log_test_loss, log_test_acc = [], []

    idx_best_loss = 0
    idx_best_acc = 0

    # Training
    for epoch in range(1, n_epochs + 1):
        print(f"\n===> Epoch {epoch}/{n_epochs}, lr: {scheduler.get_last_lr()}")

        train_loss, train_acc = train_loop(
            train_loader, model, criterion, optimizer, device
        )

        if evaluate_train:
            train_loss, train_acc = test_loop(
                train_loader, model, criterion, device
            )

        test_loss, test_acc = test_loop(
            test_loader, model, criterion, device
        )

        scheduler.step()

        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_test_loss.append(test_loss)
        log_test_acc.append(test_acc)

        if test_loss <= log_test_loss[idx_best_loss]:
            torch.save(model.state_dict(), os.path.join(save_dir, "loss_model.pth"))
            idx_best_loss = epoch - 1
            print("Saved loss-best model")

        if test_acc >= log_test_acc[idx_best_acc]:
            torch.save(model.state_dict(), os.path.join(save_dir, "acc_model.pth"))
            idx_best_acc = epoch - 1
            print("Saved acc-best model")

    # Final save
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))

    # Save logs
    np.save(os.path.join(save_dir, "log_train_loss.npy"), np.array(log_train_loss))
    np.save(os.path.join(save_dir, "log_train_acc.npy"), np.array(log_train_acc))
    np.save(os.path.join(save_dir, "log_test_loss.npy"), np.array(log_test_loss))
    np.save(os.path.join(save_dir, "log_test_acc.npy"), np.array(log_test_acc))

    # Plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(log_train_loss)
    plt.plot(log_test_loss)
    plt.title("Loss")
    plt.legend(["Train", "Test"])
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(log_train_acc)
    plt.plot(log_test_acc)
    plt.title("Accuracy")
    plt.legend(["Train", "Test"])
    plt.grid()

    plt.savefig(os.path.join(save_dir, "log.png"))
    plt.show()
