#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

from resnet_WTHaar import ResNet20


# -----------------------------
# CIFAR-10 Dataset (HF)
# -----------------------------
class HFCIFAR10(Dataset):
    def __init__(self, split, transform=None):
        self.ds = load_dataset("cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["img"]      # PIL Image
        label = self.ds[idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------
# Test loop
# -----------------------------
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    test_loss = 0.0
    test_acc = 0.0
    count = 0

    pbar = tqdm(dataloader)
    with torch.no_grad():
        for x, labels in pbar:
            x = x.to(device)
            labels = labels.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, 1)
            num_correct = (preds == labels).sum().item()

            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size
            test_acc += num_correct
            count += batch_size

            acc = num_correct / batch_size
            pbar.set_description(f"loss: {loss.item():.4f}, acc: {acc:.4f}")

    return test_loss / count, test_acc / count


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    batch_size = 128
    save_dir = "./saved"
    model_path = os.path.join(save_dir, "acc_model.pth")

    # Transforms (CIFAR-10 standard)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # Dataset + Loader
    test_dataset = HFCIFAR10(split="test", transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = ResNet20().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # Test
    test_loss, test_acc = test_loop(test_loader, model, criterion, device)
    print(f"\nTest loss: {test_loss:.6f}, acc: {test_acc:.6f}")
