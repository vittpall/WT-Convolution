import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from scipy.linalg import hadamard
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import json

# Import transforms from layers
from layers.WHT import hadamard_transform
from layers.WTHaar import haar_transform_1d, haar_transform_2d

rcParams['figure.figsize'] = (14, 10)
rcParams['font.size'] = 11

# ---------------------------
# Noise Functions
# ---------------------------


def find_min_power(x, p=2):
    """Find the smallest power of p that is >= x."""
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
# Noise Analysis Functions
# ---------------------------

def create_sample_images(dataset, device="cpu", out_dir="samples"):
    """Create sample images showing different noise types at different levels."""
    os.makedirs(out_dir, exist_ok=True)
    
    img, _ = dataset[0]
    img = img.unsqueeze(0).to(device)
    
    noise_types = {
        'Gaussian': [0.05, 0.1, 0.15, 0.2],
        'Salt & Pepper': [0.05, 0.1, 0.15, 0.2],
        'Poisson': [0.05, 0.1, 0.15, 0.2],
        'Uniform': [0.05, 0.1, 0.15, 0.2],
    }
    
    fig, axes = plt.subplots(len(noise_types) + 1, 5, figsize=(16, 14))
    
    # Original image
    axes[0, 0].imshow(img.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1))
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')
    
    for i in range(1, 5):
        axes[0, i].axis('off')
    
    # Add noise samples
    row = 1
    for noise_name, levels in noise_types.items():
        for col, level in enumerate(levels):
            if noise_name == 'Gaussian':
                noisy = add_gaussian_noise(img, std=level)
            elif noise_name == 'Salt & Pepper':
                noisy = add_salt_pepper_noise(img, prob=level)
            elif noise_name == 'Poisson':
                noisy = add_poisson_noise(img, lam=level)
            else:  # Uniform
                noisy = add_uniform_noise(img, std=level)
            
            axes[row, col].imshow(noisy.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1))
            axes[row, col].set_title(f'{noise_name}\n(level={level:.2f})', fontsize=9)
            axes[row, col].axis('off')
        row += 1
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/noise_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved noise samples to {out_dir}/noise_samples.png")


def create_noise_comparison_plots(dataset, device="cpu", out_dir="metrics", num_samples=10):
    """Create comparison plots for noise metrics across different noise types and levels."""
    os.makedirs(out_dir, exist_ok=True)
    
    noise_types = ['Gaussian', 'Salt & Pepper', 'Poisson', 'Uniform', 'Blur']
    noise_levels = np.linspace(0.02, 0.25, 8)
    
    results = {noise_type: {'psnr': [], 'ssim': [], 'mse': []} 
               for noise_type in noise_types}
    
    print(f"   Computing metrics for {num_samples} samples...")
    
    for sample_idx in range(min(num_samples, len(dataset))):
        img, _ = dataset[sample_idx]
        img = img.unsqueeze(0).to(device)
        
        for noise_type in noise_types:
            psnr_vals, ssim_vals, mse_vals = [], [], []
            
            for level in noise_levels:
                if noise_type == 'Gaussian':
                    noisy = add_gaussian_noise(img, std=level)
                elif noise_type == 'Salt & Pepper':
                    noisy = add_salt_pepper_noise(img, prob=level)
                elif noise_type == 'Poisson':
                    noisy = add_poisson_noise(img, lam=level)
                elif noise_type == 'Uniform':
                    noisy = add_uniform_noise(img, std=level)
                else:  # Blur
                    noisy = add_blur_noise(img, kernel_size=int(3 + level*10))
                
                # Metrics
                mse = F.mse_loss(noisy, img).item()
                psnr = -10 * np.log10(mse + 1e-10)
                
                mse_vals.append(mse)
                psnr_vals.append(psnr)
            
            results[noise_type]['mse'].append(mse_vals)
            results[noise_type]['psnr'].append(psnr_vals)
    
    # Average results
    for noise_type in noise_types:
        results[noise_type]['mse'] = np.mean(results[noise_type]['mse'], axis=0)
        results[noise_type]['psnr'] = np.mean(results[noise_type]['psnr'], axis=0)
    
    # Plot 1: MSE vs Noise Level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for noise_type in noise_types:
        axes[0].plot(noise_levels, results[noise_type]['mse'], marker='o', label=noise_type, linewidth=2)
    
    axes[0].set_xlabel('Noise Level', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('MSE vs Noise Level', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: PSNR vs Noise Level
    for noise_type in noise_types:
        axes[1].plot(noise_levels, results[noise_type]['psnr'], marker='s', label=noise_type, linewidth=2)
    
    axes[1].set_xlabel('Noise Level', fontsize=12)
    axes[1].set_ylabel('Peak Signal-to-Noise Ratio (dB)', fontsize=12)
    axes[1].set_title('PSNR vs Noise Level', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/noise_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics comparison to {out_dir}/noise_metrics_comparison.png")
    
    return results


def create_detailed_noise_analysis(dataset, device="cpu", out_dir="statistics", num_samples=15):
    """Detailed statistical analysis of noise robustness."""
    os.makedirs(out_dir, exist_ok=True)
    
    noise_configs = [
        ('Gaussian', [0.05, 0.1, 0.15, 0.2]),
        ('Salt & Pepper', [0.05, 0.1, 0.15, 0.2]),
        ('Poisson', [0.05, 0.1, 0.15, 0.2]),
        ('Uniform', [0.05, 0.1, 0.15, 0.2]),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    print(f"   Computing statistics for {num_samples} samples...")
    
    for idx, (noise_type, levels) in enumerate(noise_configs):
        metrics_per_level = []
        
        for sample_idx in range(min(num_samples, len(dataset))):
            img, _ = dataset[sample_idx]
            img = img.unsqueeze(0).to(device)
            
            metrics_for_sample = []
            for level in levels:
                if noise_type == 'Gaussian':
                    noisy = add_gaussian_noise(img, std=level)
                elif noise_type == 'Salt & Pepper':
                    noisy = add_salt_pepper_noise(img, prob=level)
                elif noise_type == 'Poisson':
                    noisy = add_poisson_noise(img, lam=level)
                else:  # Uniform
                    noisy = add_uniform_noise(img, std=level)
                
                mse = F.mse_loss(noisy, img).item()
                metrics_for_sample.append(mse)
            
            metrics_per_level.append(metrics_for_sample)
        
        metrics_per_level = np.array(metrics_per_level)
        means = metrics_per_level.mean(axis=0)
        stds = metrics_per_level.std(axis=0)
        
        # Plot with error bars
        axes[idx].errorbar(levels, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2, label='Mean ± Std')
        axes[idx].fill_between(levels, means - stds, means + stds, alpha=0.3)
        axes[idx].set_xlabel('Noise Level', fontsize=11)
        axes[idx].set_ylabel('MSE', fontsize=11)
        axes[idx].set_title(f'{noise_type} Noise - Statistical Analysis', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/detailed_noise_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved detailed statistics to {out_dir}/detailed_noise_statistics.png")


# ---------------------------
# Main Comparison
# ---------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    target_size = find_min_power(224)
    
    # Load dataset
    print("Loading Mini-ImageNet dataset...")
    dataset = HFMiniImageNet(
        "train",
        transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
    )
    
    # Create output directories
    out_base = "noise_robustness_analysis"
    os.makedirs(out_base, exist_ok=True)
    
    print("\n" + "="*70)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("="*70)
    
    # 1. Noise metrics comparison
    print("\n1. Generating noise metrics comparison...")
    results = create_noise_comparison_plots(
        dataset, 
        device=device, 
        out_dir=f"{out_base}/metrics",
        num_samples=10
    )
    
    # 2. Sample images with different noise types
    print("\n2. Creating sample noise visualizations...")
    create_sample_images(
        dataset,
        device=device,
        out_dir=f"{out_base}/samples"
    )
    
    # 3. Detailed statistical analysis
    print("\n3. Performing detailed statistical analysis...")
    create_detailed_noise_analysis(
        dataset,
        device=device,
        out_dir=f"{out_base}/statistics",
        num_samples=15
    )
    
    # 4. Try to load and compare models if they exist
    print("\n4. Looking for trained models...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_path)
    
    wht_model_paths = glob.glob(os.path.join(parent_dir, "WHTResNet50x3*/checkpoint*.pth"))
    wthaar_model_paths = glob.glob(os.path.join(parent_dir, "WTHaarResNet50x3*/checkpoint*.pth"))
    
    if wht_model_paths or wthaar_model_paths:
        print(f"Found WHT models: {len(wht_model_paths)}")
        print(f"Found WTHaar models: {len(wthaar_model_paths)}")
        
        # Create model comparison summary
        summary = {
            'wht_models': wht_model_paths,
            'wthaar_models': wthaar_model_paths,
            'timestamp': str(np.datetime64('today')),
            'num_dataset_samples': len(dataset),
            'dataset_size': target_size
        }
        
        with open(f"{out_base}/model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved model summary to {out_base}/model_summary.json")
    else:
        print("No trained models found. Please ensure WHTResNet50x3 and WTHaarResNet50x3 folders exist.")
    
    print("\n" + "="*70)
    print(f"Analysis complete! Results saved to: {out_base}/")
    print("="*70)


if __name__ == "__main__":
    main()
