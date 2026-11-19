#!/usr/bin/env python3
"""One script for every ResNet-50 variant on Mini-ImageNet (timm/mini-imagenet, 100 classes).

NOTE: the paper calls this dataset "Tiny-ImageNet"; the original scripts in this folder load
timm/mini-imagenet (train 50k / validation 10k / test 5k, 224 px). This script keeps that dataset.

Protocol (identical for every --kind): RandomResizedCrop(224)+flip / Resize(256)+CenterCrop(224),
ImageNet mean/std, SGD lr 0.1 momentum 0.9 wd 1e-4, batch 128, 90 epochs, StepLR(30, 0.1).
Architecture: torchvision-layout ResNet-50 v1.5 (from ../WTHaarResNet50x3/WTHaarResNet.py), with the
3x3 conv2 of every non-first bottleneck in each stage replaced by the transform layer
(AvgPool before it when strided), exactly as in the original HT/Haar scripts. --kind conv uses
torchvision.models.resnet50(num_classes=100) (standard ResNet-50 v1.5).
Reported: final-epoch val + test, and test acc of the best-VALIDATION epoch (no test selection).
Resumes automatically from <out>/<tag>_ckpt.pth (24 h walltime).
"""
import argparse, importlib.util, json, os, random, sys, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument('--kind', required=True, choices=['conv', 'ht', 'haar', 'lhaar'])
p.add_argument('--levels', type=int, default=None)
p.add_argument('--pods', type=int, default=3)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--epochs', type=int, default=90)
p.add_argument('-b', '--batch-size', type=int, default=128)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--workers', type=int, default=16)
p.add_argument('--out', default=os.path.join(HERE, 'results'))
p.add_argument('--max-batches', type=int, default=0, help='smoke test only')
args = p.parse_args()

tag = f"{args.kind}{'' if args.levels is None else f'_L{args.levels}'}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
res_path, ckpt_path = os.path.join(args.out, tag + '.json'), os.path.join(args.out, tag + '_ckpt.pth')
if os.path.exists(res_path) and not args.max_batches:
    print('done already:', res_path); raise SystemExit

# ---- model: original ResNet-50 file, mixer swapped for the unified TransformConv2D ----
sys.path.insert(0, os.path.join(HERE, '..', 'WTHaarResNet50x3'))
import WTHaarResNet as R  # noqa: E402
spec = importlib.util.spec_from_file_location('exp_layers', os.path.join(HERE, '..', '..', 'CIFAR10', 'exp', 'layers.py'))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)


def mixer(h, w, cin, cout, pods=3, residual=False):
    return L.TransformConv2D(h, w, cin, cout, pods=args.pods, residual=residual,
                             kind=args.kind, levels=args.levels)


R.HWTConv2D = mixer
if args.kind == 'conv':
    import torchvision
    model = torchvision.models.resnet50(num_classes=100)  # standard ResNet-50 v1.5
else:
    model = R.wthaar_resnet50(num_classes=100)

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(dev)
n_params = sum(q.numel() for q in model.parameters())


class HF(Dataset):
    def __init__(self, split, tf):
        from datasets import load_dataset
        self.ds, self.tf = load_dataset('timm/mini-imagenet', split=split), tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return self.tf(r['image'].convert('RGB')), r['label']


norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_tr = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), norm])
tf_te = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
tr_ld = DataLoader(HF('train', tf_tr), args.batch_size, shuffle=True, drop_last=True, **kw)
va_ld = DataLoader(HF('validation', tf_te), 256, **kw)
te_ld = DataLoader(HF('test', tf_te), 256, **kw)

opt = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.StepLR(opt, 30, 0.1)
crit = nn.CrossEntropyLoss()


def evaluate(ld):
    model.eval(); c1 = c5 = n = 0; loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(ld):
            if args.max_batches and i >= args.max_batches: break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            o = model(x); loss += crit(o, y).item() * len(y)
            t5 = o.topk(5, 1).indices
            c1 += (t5[:, 0] == y).sum().item(); c5 += (t5 == y[:, None]).any(1).sum().item(); n += len(y)
    return loss / n, 100 * c1 / n, 100 * c5 / n


log = {k: [] for k in ('train_loss', 'val_loss', 'val_acc1', 'val_acc5', 'test_acc1', 'test_acc5')}
start, best_val, minutes = 0, -1.0, 0.0
if os.path.exists(ckpt_path) and not args.max_batches:
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    start, log, best_val, minutes = ck['epoch'], ck['log'], ck['best_val'], ck['minutes']
    torch.set_rng_state(ck['rng'].cpu()); print(f'resumed {tag} at epoch {start}')

print(f'{tag}: {n_params:,} params, device {dev}', flush=True)
for ep in range(start, args.epochs):
    t0 = time.time(); model.train(); tl = nb = 0
    for i, (x, y) in enumerate(tr_ld):
        if args.max_batches and i >= args.max_batches: break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        l = crit(model(x), y); l.backward(); opt.step()
        tl += l.item(); nb += 1
    sched.step()
    vl, v1, v5 = evaluate(va_ld); _, t1, t5 = evaluate(te_ld)
    for k, v in zip(log, (tl / nb, vl, v1, v5, t1, t5)): log[k].append(v)
    minutes += (time.time() - t0) / 60
    if v1 > best_val and not args.max_batches:
        best_val = v1; torch.save(model.state_dict(), os.path.join(args.out, tag + '_bestval.pth'))
    print(f'[{tag}] ep {ep+1}/{args.epochs} loss {tl/nb:.3f} val {v1:.2f}/{v5:.2f} test {t1:.2f}/{t5:.2f} '
          f'({minutes:.0f} min)', flush=True)
    if not args.max_batches:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        log=log, best_val=best_val, minutes=minutes, rng=torch.get_rng_state()), ckpt_path)

ib = int(np.argmax(log['val_acc1']))
out = dict(kind=args.kind, levels=args.levels, pods=args.pods, seed=args.seed, params=n_params, epochs=args.epochs,
           final_val_acc1=log['val_acc1'][-1], final_test_acc1=log['test_acc1'][-1],
           final_test_acc5=log['test_acc5'][-1], bestval_epoch=ib + 1,
           bestval_test_acc1=log['test_acc1'][ib], bestval_test_acc5=log['test_acc5'][ib],
           minutes=minutes, log=log)
if args.max_batches:
    print('smoke ok', {k: v for k, v in out.items() if k != 'log'})
else:
    torch.save(model.state_dict(), os.path.join(args.out, tag + '_final.pth'))
    json.dump(out, open(res_path, 'w')); os.remove(ckpt_path); print('saved', res_path)
