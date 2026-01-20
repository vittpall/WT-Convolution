import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from scipy.linalg import hadamard
from datasets import load_dataset

# ---------------------------
# Haar Transform
# ---------------------------

def haar_transform_1d(u, axis=-1, inverse=False):
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m

    x = u.clone()
    norm = 1.0 / np.sqrt(2)

    if not inverse:
        length = n
        for _ in range(m):
            half = length // 2
            temp = x[..., :length].clone()
            x[..., :half] = norm * (temp[..., ::2] + temp[..., 1::2])
            x[..., half:length] = norm * (temp[..., ::2] - temp[..., 1::2])
            length = half
    else:
        length = 2
        for _ in range(m):
            half = length // 2
            temp = x[..., :length].clone()
            a = temp[..., :half]
            d = temp[..., half:length]
            x[..., :length:2] = norm * (a + d)
            x[..., 1:length:2] = norm * (a - d)
            length *= 2

    if axis != -1:
        x = torch.transpose(x, -1, axis)

    return x


def haar_transform_2d(x, inverse=False):
    x = haar_transform_1d(x, axis=-1, inverse=inverse)
    x = haar_transform_1d(x, axis=-2, inverse=inverse)
    return x


# ---------------------------
# Hadamard Transform
# ---------------------------

def hadamard_transform(u, axis=-1):
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m

    H = torch.tensor(hadamard(n), dtype=u.dtype, device=u.device)
    y = (u @ H.T) / np.sqrt(n)

    if axis != -1:
        y = torch.transpose(y, -1, axis)

    return y


def add_noise(x, std=0.1):
    return torch.clamp(x + std * torch.randn_like(x), 0.0, 1.0)


def soft_threshold(x, T):
    return torch.sign(x) * torch.relu(torch.abs(x) - T)


def haar_denoise(x, T):
    y = haar_transform_2d(x, inverse=False)
    print(y)
    y = soft_threshold(y, T)
    return haar_transform_2d(y, inverse=True)


def wht_denoise(x, T):
    y = hadamard_transform(x, axis=-1)
    y = hadamard_transform(y, axis=-2)
    y = soft_threshold(y, T)
    y = hadamard_transform(y, axis=-1)
    y = hadamard_transform(y, axis=-2)
    return y


def find_min_power(x, p=2):
    y = 1
    while y < x:
        y *= p
    return y


# ---------------------------
# Dataset
# ---------------------------

class HFMiniImageNet(Dataset):
    def __init__(self, split, transform=None):
        self.ds = load_dataset("timm/mini-imagenet", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        label = self.ds[idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------
# Comparison
# ---------------------------

def compare_transforms(dataset, device="cpu", out_dir="transform_comparison"):
    os.makedirs(out_dir, exist_ok=True)

    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)

    noisy = add_noise(img, std=0.1)
    T = 0.1

    with torch.no_grad():
        recon_haar = haar_denoise(noisy, T)
        recon_wht = wht_denoise(noisy, T)

    mse_noisy = F.mse_loss(noisy, img).item()
    mse_haar = F.mse_loss(recon_haar, img).item()
    mse_wht = F.mse_loss(recon_wht, img).item()

    save_image(img, f"{out_dir}/original.png")
    save_image(noisy, f"{out_dir}/noisy.png")
    save_image(recon_haar, f"{out_dir}/haar_denoised.png")
    save_image(recon_wht, f"{out_dir}/wht_denoised.png")

    with open(f"{out_dir}/metrics.txt", "w") as f:
        f.write(f"Noisy MSE : {mse_noisy:.8e}\n")
        f.write(f"Haar  MSE : {mse_haar:.8e}\n")
        f.write(f"WHT   MSE : {mse_wht:.8e}\n")

    print("Results saved to:", out_dir)
    print(f"Noisy MSE : {mse_noisy:.8e}")
    print(f"Haar  MSE : {mse_haar:.8e}")
    print(f"WHT   MSE : {mse_wht:.8e}")


# ---------------------------
# Main
# ---------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_size = find_min_power(224)

    dataset = HFMiniImageNet(
        "train",
        transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
    )

    compare_transforms(dataset, device=device)


if __name__ == "__main__":
    main()
