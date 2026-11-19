#!/usr/bin/env python3
"""Train one (config, seed) on CIFAR-10 with a single, shared protocol.

Protocol (identical for every config):
  torchvision CIFAR-10, pad-4 random crop + flip, per-channel mean/std
  (0.4914, 0.4822, 0.4465) / (0.2470, 0.2435, 0.2616);
  SGD lr 0.1, momentum 0.9, wd 1e-4 on weights only, batch 128,
  200 epochs, lr x0.1 at epochs 82/122/163.
Reported metric: FINAL-epoch test accuracy (no checkpoint selection on test).
best-epoch test accuracy is logged for reference only.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision import transforms
from models import build

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '..', 'data'))
p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results'))
p.add_argument('--max-batches', type=int, default=0, help='smoke test only')
p.add_argument('--workers', type=int, default=3)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
tag = f'{args.config}_s{args.seed}'
res_path = os.path.join(args.out, tag + '.json')
if os.path.exists(res_path) and not args.max_batches:
    print('done already:', res_path); raise SystemExit

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(), norm])
te = transforms.Compose([transforms.ToTensor(), norm])
train_ld = DataLoader(dsets.CIFAR10(args.data, True, tr, download=False), 128, shuffle=True,
                      num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
test_ld = DataLoader(dsets.CIFAR10(args.data, False, te, download=False), 256, shuffle=False,
                     num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

model = build(args.config).to(dev)
n_params = sum(q.numel() for q in model.parameters())
wd, nwd = [], []
for n, q in model.named_parameters():
    (nwd if ('bias' in n or 'theta' in n) else wd).append(q)
opt = torch.optim.SGD([{'params': wd, 'weight_decay': 1e-4}, {'params': nwd, 'weight_decay': 0}],
                      lr=0.1, momentum=0.9)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [82, 122, 163], 0.1)
crit = nn.CrossEntropyLoss()


def evaluate():
    model.eval(); correct = tot = 0; loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_ld):
            if args.max_batches and i >= args.max_batches: break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            o = model(x); loss += crit(o, y).item() * len(y)
            correct += (o.argmax(1) == y).sum().item(); tot += len(y)
    return loss / tot, 100.0 * correct / tot


log = {'train_loss': [], 'test_loss': [], 'test_acc': []}
t0 = time.time()
for ep in range(args.epochs):
    model.train(); tl = 0.0; nb = 0
    for i, (x, y) in enumerate(train_ld):
        if args.max_batches and i >= args.max_batches: break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        l = crit(model(x), y); l.backward(); opt.step()
        tl += l.item(); nb += 1
    sched.step()
    vl, va = evaluate()
    log['train_loss'].append(tl / nb); log['test_loss'].append(vl); log['test_acc'].append(va)
    if ep % 10 == 0 or ep == args.epochs - 1:
        print(f'[{tag}] ep {ep+1}/{args.epochs} train_loss {tl/nb:.4f} test_acc {va:.2f} '
              f'({(time.time()-t0)/60:.1f} min)', flush=True)

out = dict(config=args.config, seed=args.seed, epochs=args.epochs, params=n_params,
           final_acc=log['test_acc'][-1], best_acc_ref=max(log['test_acc']),
           minutes=(time.time() - t0) / 60, log=log)
if args.max_batches:
    print('smoke ok', {k: v for k, v in out.items() if k != 'log'})
else:
    torch.save(model.state_dict(), os.path.join(args.out, tag + '.pth'))
    with open(res_path, 'w') as f:
        json.dump(out, f)
    print('saved', res_path)
