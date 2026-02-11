import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Callable, Tuple
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from torchvision import transforms, datasets
from datasets import load_dataset

from WHTResNet import wht_resnet50
from WTHaarResNet import wthaar_resnet50



class HFMiniImageNet(Dataset):
    def __init__(self, split, transform=None):
        self.ds = load_dataset("timm/mini-imagenet", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        label = self.ds[idx]["label"]
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        return img, label


class Config:
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    NUM_BATCHES = 100
    GPU_ID = 0
    IMAGENET_DIR = '/data/ImageNet'
    SAVE_DIR = './results'
    
    GAUSSIAN_STDS = [0.05, 0.10, 0.15, 0.20, 0.25]
    SALT_PEPPER_PROBS = [0.02, 0.05, 0.10, 0.15, 0.20]
    POISSON_LAMBDAS = [0.2, 0.5, 1.0, 2.0, 5.0]
    BLUR_KERNELS = [3, 5, 7, 9, 11]



class NoiseGenerator:
    
    @staticmethod
    def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)
    
    @staticmethod
    def salt_pepper_noise(x: torch.Tensor, prob: float) -> torch.Tensor:
        noisy = x.clone()
        mask = torch.rand_like(x) < prob
        noisy[mask] = torch.randint(0, 2, (mask.sum().item(),), 
                                   dtype=torch.float32).to(x.device)
        return noisy
    
    @staticmethod
    def poisson_noise(x: torch.Tensor, lam: float) -> torch.Tensor:
        x_scaled = torch.clamp(x * lam, min=0)
        noisy = torch.poisson(x_scaled) / lam
        return torch.clamp(noisy, 0, 1)
    
    @staticmethod
    def gaussian_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        from torchvision.transforms.functional import gaussian_blur
        return gaussian_blur(x, kernel_size)
    
    @staticmethod
    def motion_blur_simulation(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        result = x.clone()
        for _ in range(2):
            result = NoiseGenerator.gaussian_blur(result, kernel_size)
        return result



def load_models(gpu_id: int) -> Dict[str, nn.Module]:
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on device: {device}\n")
    
    models = {}
    
    print("Loading WHT ResNet50...")
    wht_model = torch.nn.DataParallel(wht_resnet50(num_classes=100)).cuda()
    checkpoint_wht = torch.load('../WHTResNet50x3/saved/wht_resnet50_acc1_model.pth')
    wht_model.load_state_dict(checkpoint_wht)
    wht_model.eval()
    models['WHT'] = wht_model
    print("✓ WHT model loaded\n")
    
    print("Loading WTHaar ResNet50...")
    wthaar_model = torch.nn.DataParallel(wthaar_resnet50(num_classes=100)).cuda()
    checkpoint_wthaar = torch.load('../WTHaarResNet50x3/saved/wthaar_resnet50_acc1_model.pth')
    wthaar_model.load_state_dict(checkpoint_wthaar)
    wthaar_model.eval()
    models['WTHaar'] = wthaar_model
    print("✓ WTHaar model loaded\n")
    
    return models


def get_data_loader(batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    print("Loading Mini-ImageNet validation dataset...")
    val_dataset = HFMiniImageNet(split='validation', transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    print(f"✓ Loaded {len(val_dataset)} validation samples\n")
    
    return val_loader



def evaluate_model(model: nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  noise_fn: Callable,
                  num_batches: int,
                  description: str = "") -> float:
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, total=num_batches, desc=description, leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if batch_idx >= num_batches:
                break
            
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            try:
                noisy_inputs = noise_fn(inputs)
            except Exception as e:
                print(f"Warning: Noise function failed - {e}, using clean inputs")
                noisy_inputs = inputs
            
            outputs = model(noisy_inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def run_robustness_evaluation(models: Dict[str, nn.Module],
                             dataloader: torch.utils.data.DataLoader,
                             config: Config) -> Dict[str, Dict]:
    results = {model_name: {} for model_name in models.keys()}
    
    noise_configs = {
        'Gaussian Noise': [
            (f'σ={std:.2f}', 
             lambda x, s=std: NoiseGenerator.gaussian_noise(x, s))
            for std in config.GAUSSIAN_STDS
        ],
        'Salt & Pepper': [
            (f'p={prob:.2f}', 
             lambda x, p=prob: NoiseGenerator.salt_pepper_noise(x, p))
            for prob in config.SALT_PEPPER_PROBS
        ],
        'Poisson Noise': [
            (f'λ={lam:.1f}', 
             lambda x, l=lam: NoiseGenerator.poisson_noise(x, l))
            for lam in config.POISSON_LAMBDAS
        ],
        'Gaussian Blur': [
            (f'k={k}', 
             lambda x, k=k: NoiseGenerator.gaussian_blur(x, k))
            for k in config.BLUR_KERNELS
        ],
    }
    
    print("\n" + "="*70)
    print("EVALUATING CLEAN BASELINE")
    print("="*70 + "\n")
    
    for model_name, model in models.items():
        acc = evaluate_model(model, dataloader, lambda x: x, config.NUM_BATCHES,
                           f"Clean | {model_name}")
        results[model_name]['Clean'] = acc
        print(f"{model_name:12s} Clean accuracy: {acc:6.2f}%")
    
    for noise_type, noise_list in noise_configs.items():
        print("\n" + "="*70)
        print(f"EVALUATING {noise_type.upper()}")
        print("="*70 + "\n")
        
        for noise_label, noise_fn in noise_list:
            print(f"Testing: {noise_type} - {noise_label}")
            for model_name, model in models.items():
                acc = evaluate_model(model, dataloader, noise_fn, 
                                   config.NUM_BATCHES,
                                   f"{noise_type} {noise_label} | {model_name}")
                results[model_name][f'{noise_type} {noise_label}'] = acc
                print(f"  {model_name:12s}: {acc:6.2f}%")
    
    return results



def plot_robustness_comparison(results: Dict[str, Dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    noise_types = list(set(
        noise_key.split()[0:2] for model_results in results.values() 
        for noise_key in model_results.keys() if noise_key != 'Clean'
    ))
    
    print("\nGenerating plots...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(results.keys())
    
    summary_data = {model: {} for model in model_names}
    summary_data['Clean'] = {
        model: results[model]['Clean'] for model in model_names
    }
    
    for noise_type_name in ['Gaussian Noise', 'Salt & Pepper', 'Poisson Noise', 'Gaussian Blur']:
        averages = {}
        for model in model_names:
            noise_accs = [acc for key, acc in results[model].items() 
                         if key.startswith(noise_type_name)]
            if noise_accs:
                averages[model] = np.mean(noise_accs)
        if averages:
            summary_data[noise_type_name] = averages
    
    x = np.arange(len(summary_data))
    width = 0.35
    
    for idx, model_name in enumerate(model_names):
        values = [summary_data[category][model_name] for category in summary_data.keys()]
        offset = (idx - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, alpha=0.85)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Comparison: WHT vs WTHaar ResNet50', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_data.keys(), rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_summary_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/01_summary_comparison.png")
    
    noise_categories = ['Gaussian Noise', 'Salt & Pepper', 'Poisson Noise', 'Gaussian Blur']
    
    for plot_idx, noise_category in enumerate(noise_categories, 2):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        noise_labels = []
        noise_data = {model: [] for model in model_names}
        
        for key in sorted(results[model_names[0]].keys()):
            if key.startswith(noise_category):
                noise_labels.append(key.replace(f'{noise_category} ', ''))
                for model in model_names:
                    noise_data[model].append(results[model][key])
        
        if not noise_labels:
            continue
        
        x = np.arange(len(noise_labels))
        width = 0.35
        
        for idx, model_name in enumerate(model_names):
            offset = (idx - len(model_names)/2 + 0.5) * width
            ax.bar(x + offset, noise_data[model_name], width, label=model_name, alpha=0.85)
        
        ax.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{noise_category} Robustness Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{plot_idx:02d}_{noise_category.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/{plot_idx:02d}_{noise_category.lower().replace(' ', '_')}.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for plot_idx, noise_category in enumerate(noise_categories):
        ax = axes[plot_idx]
        
        noise_labels = []
        noise_data = {model: [] for model in model_names}
        
        for key in sorted(results[model_names[0]].keys()):
            if key.startswith(noise_category):
                noise_labels.append(key.replace(f'{noise_category} ', ''))
                for model in model_names:
                    noise_data[model].append(results[model][key])
        
        if not noise_labels:
            continue
        
        for model_name in model_names:
            ax.plot(noise_labels, noise_data[model_name], marker='o', linewidth=2.5,
                   markersize=8, label=model_name, markerfacecolor='none', markeredgewidth=2)
        
        ax.set_xlabel('Noise Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{noise_category}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(noise_labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_line_plots_combined.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/06_line_plots_combined.png")
    
    plt.close('all')


def print_robustness_summary(results: Dict[str, Dict]):
    print("\n" + "="*70)
    print("ROBUSTNESS SUMMARY")
    print("="*70 + "\n")
    
    for model_name, model_results in results.items():
        clean_acc = model_results['Clean']
        all_accs = np.array(list(model_results.values()))
        
        print(f"{model_name}:")
        print(f"  Clean Accuracy:      {clean_acc:6.2f}%")
        print(f"  Average Accuracy:    {np.mean(all_accs):6.2f}%")
        print(f"  Min Accuracy:        {np.min(all_accs):6.2f}%")
        print(f"  Max Accuracy:        {np.max(all_accs):6.2f}%")
        print(f"  Std Dev:             {np.std(all_accs):6.2f}%")
        print(f"  Robustness Drop:     {clean_acc - np.min(all_accs):6.2f}%")
        print()



def main():
    parser = argparse.ArgumentParser(
        description='Compare WHT and WTHaar ResNet50 robustness to noise'
    )
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--num-batches', type=int, default=Config.NUM_BATCHES)
    parser.add_argument('--gpu', type=int, default=Config.GPU_ID)
    parser.add_argument('--save-dir', type=str, default=Config.SAVE_DIR)
    
    args = parser.parse_args()
    
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_WORKERS = args.num_workers
    Config.NUM_BATCHES = args.num_batches
    Config.GPU_ID = args.gpu
    Config.SAVE_DIR = args.save_dir
    
    print("\n" + "="*70)
    print("ROBUSTNESS COMPARISON: WHT vs WTHaar ResNet50")
    print("="*70 + "\n")
    
    models = load_models(args.gpu)
    
    dataloader = get_data_loader(args.batch_size, args.num_workers)
    
    results = run_robustness_evaluation(models, dataloader, Config)
    
    print_robustness_summary(results)
    
    os.makedirs(args.save_dir, exist_ok=True)
    results_json = {k: {kk: float(vv) for kk, vv in v.items()} 
                   for k, v in results.items()}
    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Saved: {args.save_dir}/results.json")
    
    np.save(f'{args.save_dir}/results.npy', results)
    print(f"✓ Saved: {args.save_dir}/results.npy")
    
    plot_robustness_comparison(results, args.save_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
    