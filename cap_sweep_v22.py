#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cap Sweep (Option C) with 5 Seeds
===============================================

This script runs the cap sweep experiment with multiple random seeds and reports
mean ± std accuracy for each method at each k_train.

Hard pair by default:
  vehicles_all -> animals_all

Per model:
  - NONE baseline: linear head on raw D-dim features
  - RANDOM: random orthonormal projection to k_train dims
  - SPECTRAL: source KL subspace truncated to k_train dims

For each seed:
  - sample source train set (Ns) to estimate KL basis
  - sample target train set (Nt) to train head with val split
  - sample target test set (Ntest) to evaluate

Outputs:
  - outputs_review/cap_sweep_results_mean_std.csv   (aggregated mean/std)
  - outputs_review/cap_sweep_results_per_seed.csv   (raw per-seed)
  - outputs_review/cap_sweep_plot_mean_std.png      (plot with ±1 std bands)

Recommended run (5 seeds):
  python3 cap_sweep_v22.py \
    --v22_csv scaling_results_multibackbone_v22.csv \
    --src vehicles_all --tgt animals_all \
    --Ns 2000 --Nt 100 --Ntest 2000 \
    --caps 32,64,99,kstar \
    --epochs 30 \
    --seeds 20260119,20260218,20260301,20260315,20260329


"""

import os
import csv
import argparse
import hashlib
import warnings
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT = True
except Exception:
    plt = None
    PLOT = False


# ----------------------------
# Domains (CIFAR-10 label ids)
# ----------------------------
DOMAINS = {
    "vehicles_all": [0, 1, 8, 9],
    "animals_all":  [2, 3, 4, 5, 6, 7],
}

# ----------------------------
# Deterministic hashing
# ----------------------------
def stable_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)

def stable_hash_tuple(items) -> int:
    return stable_hash("|".join(str(x) for x in items))

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Backbones
# ----------------------------
def build_backbone(model_name: str, device: torch.device) -> Tuple[nn.Module, int]:
    m = model_name.lower().strip()
    if m == "resnet18":
        w = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=w)
        net.fc = nn.Identity()
        return net.to(device).eval(), 512
    if m == "resnet50":
        w = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=w)
        net.fc = nn.Identity()
        return net.to(device).eval(), 2048
    raise ValueError(f"Unsupported model: {model_name}")

# ----------------------------
# Data transforms
# ----------------------------
def imagenet_like_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def subset_indices_for_classes(labels_np: np.ndarray, class_list: List[int], limit: int, seed: int) -> List[int]:
    idx = np.where(np.isin(labels_np, np.array(class_list, dtype=np.int64)))[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return idx[:limit].tolist()

@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: torch.device, limit: int | None = None):
    feats, ys = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        h = backbone(x)
        feats.append(h.cpu())
        ys.append(y.cpu())
        n += x.size(0)
        if limit is not None and n >= limit:
            break
    X = torch.cat(feats, dim=0)
    Y = torch.cat(ys, dim=0)
    if limit is not None:
        X = X[:limit]
        Y = Y[:limit]
    return X, Y

# ----------------------------
# Covariance + KL basis
# ----------------------------
def isotropic_shrunk_cov(X: torch.Tensor, shrinkage: float = 0.1) -> torch.Tensor:
    X = X.to(torch.float64)
    X = X - X.mean(dim=0, keepdim=True)
    N, D = X.shape
    if N < 2:
        return torch.eye(D, dtype=torch.float64)
    Sigma = (X.T @ X) / (N - 1)
    a = float(np.clip(shrinkage, 0.0, 1.0))
    tr = torch.trace(Sigma)
    return (1.0 - a) * Sigma + a * (tr / D) * torch.eye(D, dtype=torch.float64)

@torch.no_grad()
def kl_basis_eigh(X_src: torch.Tensor, k: int, shrinkage: float = 0.1) -> torch.Tensor:
    Sigma = isotropic_shrunk_cov(X_src, shrinkage=shrinkage)
    evals, evecs = torch.linalg.eigh(Sigma)
    order = torch.argsort(evals, descending=True)
    U = evecs[:, order]  # [D, D]
    return U[:, :k].contiguous().to(torch.float32)

# ----------------------------
# Random orthonormal basis
# ----------------------------
@torch.no_grad()
def random_orthonormal_basis(D: int, k: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = torch.from_numpy(rng.standard_normal((D, k))).to(torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q.contiguous()

# ----------------------------
# Head training
# ----------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

def split_train_val(n: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return tr_idx, val_idx

def train_head(X_tr, y_tr, X_val, y_val, in_dim, n_classes, device, epochs, batch_size, lr, wd):
    head = LinearHead(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_val = X_val.to(device); y_val = y_val.to(device)

    best_val = -1.0
    best_state = None

    n = X_tr.shape[0]
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = head(X_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(X_val).argmax(dim=1) == y_val).float().mean().item()
        if val_acc > best_val:
            best_val = float(val_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best_val

@torch.no_grad()
def accuracy(head, X, y, device):
    head.eval()
    X = X.to(device); y = y.to(device)
    pred = head(X).argmax(dim=1)
    return float((pred == y).float().mean().item())

# ----------------------------
# CSV helper
# ----------------------------
def load_kstar_for_pair(v22_csv: str, model_name: str, src: str, tgt: str, k_field="k_star_median") -> int:
    with open(v22_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("model","").strip() == model_name and
                r.get("source","").strip() == src and
                r.get("target","").strip() == tgt):
                return int(round(float(r[k_field])))
    raise KeyError(f"Missing row for {model_name} {src}->{tgt} in {v22_csv}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v22_csv", type=str, default="scaling_results_multibackbone_v22.csv")
    ap.add_argument("--src", type=str, default="vehicles_all")
    ap.add_argument("--tgt", type=str, default="animals_all")
    ap.add_argument("--models", type=str, default="resnet18,resnet50")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])

    ap.add_argument("--Ns", type=int, default=2000, help="Source samples (train) for basis estimation")
    ap.add_argument("--Nt", type=int, default=100, help="Target samples (train) for head training")
    ap.add_argument("--Ntest", type=int, default=2000, help="Target samples (test) for evaluation")

    ap.add_argument("--caps", type=str, default="32,64,99,kstar",
                    help="Comma list of caps. Use 'kstar' token to include k* from CSV.")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--shrinkage", type=float, default=0.1)

    ap.add_argument("--seeds", type=str, default="20260218",
                    help="Comma-separated seeds, e.g. 20260119,20260218,20260301,20260315,20260329")
    ap.add_argument("--out_dir", type=str, default="outputs_review")

    args = ap.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seed_list) < 1:
        raise ValueError("Provide at least one seed in --seeds")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    src = args.src.strip()
    tgt = args.tgt.strip()
    if src not in DOMAINS or tgt not in DOMAINS:
        raise ValueError("src/tgt must be in DOMAINS dict inside script.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv_mean = os.path.join(args.out_dir, "cap_sweep_results_mean_std.csv")
    out_csv_seed = os.path.join(args.out_dir, "cap_sweep_results_per_seed.csv")
    out_png = os.path.join(args.out_dir, "cap_sweep_plot_mean_std.png")

    # datasets
    tfm = imagenet_like_transform()
    ds_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    ds_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    y_train_np = np.array(ds_train.targets, dtype=np.int64)
    y_test_np  = np.array(ds_test.targets, dtype=np.int64)

    # label remap for target (within-target classification)
    tgt_classes_sorted = sorted(set(DOMAINS[tgt]))
    remap = {c:i for i,c in enumerate(tgt_classes_sorted)}
    C = len(tgt_classes_sorted)

    def remap_labels(y_raw: torch.Tensor) -> torch.Tensor:
        return torch.tensor([remap[int(v)] for v in y_raw.tolist()], dtype=torch.long)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    # Per-seed raw rows
    per_seed_rows = []

    # Accumulator for mean/std
    # key: (model, k_train, method) -> list of accuracies
    acc = defaultdict(list)
    # also keep none per model
    none_acc = defaultdict(list)

    # Parse caps template
    cap_tokens = [t.strip().lower() for t in args.caps.split(",") if t.strip()]

    for model_name in model_list:
        backbone, D = build_backbone(model_name, device)

        # k* from CSV for this model and pair
        k_star = load_kstar_for_pair(args.v22_csv, model_name, src, tgt, k_field="k_star_median")
        k_star = max(1, min(k_star, D))

        # Make numeric caps for this model
        caps = []
        for token in cap_tokens:
            if token == "kstar":
                caps.append(k_star)
            else:
                caps.append(int(token))
        caps = sorted(set(max(1, min(int(c), D)) for c in caps))

        print("\n" + "="*70)
        print(f"{model_name}: D={D} | pair {src}->{tgt} | k*={k_star} | caps={caps}")
        print(f"Seeds: {seed_list}")
        print("="*70)

        for seed in seed_list:
            set_seed(seed)

            # SOURCE basis set (train)
            seed_src = stable_hash_tuple((seed, model_name, "train", src, args.Ns))
            idx_src = subset_indices_for_classes(y_train_np, DOMAINS[src], args.Ns, seed=int(seed_src))
            loader_src = DataLoader(Subset(ds_train, idx_src), batch_size=128, shuffle=False, num_workers=0)
            Xs, _ = extract_features(backbone, loader_src, device, limit=len(idx_src))

            # TARGET train (Nt)
            seed_ttr = stable_hash_tuple((seed, model_name, "train", tgt, args.Nt))
            idx_ttr = subset_indices_for_classes(y_train_np, DOMAINS[tgt], args.Nt, seed=int(seed_ttr))
            loader_ttr = DataLoader(Subset(ds_train, idx_ttr), batch_size=128, shuffle=False, num_workers=0)
            Xt_tr, yt_tr_raw = extract_features(backbone, loader_ttr, device, limit=len(idx_ttr))
            yt_tr = remap_labels(yt_tr_raw)

            # TARGET test (Ntest) (may truncate due to dataset size)
            seed_tte = stable_hash_tuple((seed, model_name, "test", tgt, args.Ntest))
            idx_tte = subset_indices_for_classes(y_test_np, DOMAINS[tgt], args.Ntest, seed=int(seed_tte))
            if len(idx_tte) < args.Ntest:
                warnings.warn(f"seed={seed} {model_name} test:{tgt} only has {len(idx_tte)} samples (requested {args.Ntest}).")
            loader_tte = DataLoader(Subset(ds_test, idx_tte), batch_size=128, shuffle=False, num_workers=0)
            Xt_te, yt_te_raw = extract_features(backbone, loader_tte, device, limit=len(idx_tte))
            yt_te = remap_labels(yt_te_raw)

            # Train/val split
            tr_idx, val_idx = split_train_val(
                Xt_tr.shape[0], args.val_frac,
                seed=stable_hash_tuple((seed, model_name, src, tgt, "valsplit", args.Nt))
            )
            Xt_fit, yt_fit = Xt_tr[tr_idx], yt_tr[tr_idx]
            Xt_val, yt_val = Xt_tr[val_idx], yt_tr[val_idx]

            # NONE baseline once (full D)
            head_none, _ = train_head(Xt_fit, yt_fit, Xt_val, yt_val, D, C, device,
                                      args.epochs, args.batch_size, args.lr, args.wd)
            acc_none = accuracy(head_none, Xt_te, yt_te, device)
            none_acc[model_name].append(acc_none)

            # KL basis up to k_star (capped by Ns-1)
            max_k_basis = max(1, min(D, int(Xs.shape[0]) - 1))
            k_basis = min(k_star, max_k_basis)
            U_full = kl_basis_eigh(Xs, k=k_basis, shrinkage=args.shrinkage)  # [D, k_basis]

            for k_cap in caps:
                k_train = min(k_cap, k_basis)
                if k_train < 1:
                    continue

                # RANDOM projector
                U_rand = random_orthonormal_basis(D, k_train, seed=stable_hash_tuple((seed, model_name, src, tgt, "rand", k_train)))
                with torch.no_grad():
                    Z_fit_r = (Xt_fit.to(device) @ U_rand.to(device)).cpu()
                    Z_val_r = (Xt_val.to(device) @ U_rand.to(device)).cpu()
                    Z_te_r  = (Xt_te.to(device) @ U_rand.to(device)).cpu()
                head_r, _ = train_head(Z_fit_r, yt_fit, Z_val_r, yt_val, k_train, C, device,
                                       args.epochs, args.batch_size, args.lr, args.wd)
                a_r = accuracy(head_r, Z_te_r, yt_te, device)

                # SPECTRAL projector
                U_k = U_full[:, :k_train]
                with torch.no_grad():
                    Z_fit = (Xt_fit.to(device) @ U_k.to(device)).cpu()
                    Z_val = (Xt_val.to(device) @ U_k.to(device)).cpu()
                    Z_te  = (Xt_te.to(device) @ U_k.to(device)).cpu()
                head_s, _ = train_head(Z_fit, yt_fit, Z_val, yt_val, k_train, C, device,
                                       args.epochs, args.batch_size, args.lr, args.wd)
                a_s = accuracy(head_s, Z_te, yt_te, device)

                # record
                acc[(model_name, k_train, "random")].append(a_r)
                acc[(model_name, k_train, "spectral")].append(a_s)

                per_seed_rows.append({
                    "seed": seed,
                    "model": model_name,
                    "D": D,
                    "source": src,
                    "target": tgt,
                    "Ns_source": int(args.Ns),
                    "Nt_target_train": int(args.Nt),
                    "Ntest_target": int(len(idx_tte)),
                    "k_star_csv": int(k_star),
                    "k_basis_used": int(k_basis),
                    "k_train": int(k_train),
                    "acc_none": float(acc_none),
                    "acc_random": float(a_r),
                    "acc_spectral": float(a_s),
                    "gain_spec_minus_rand": float(a_s - a_r),
                    "gain_spec_minus_none": float(a_s - acc_none),
                })

            print(f"  seed={seed} none={acc_none:.4f} | "
                  f"spec@{caps[-1]}={acc[(model_name, min(caps[-1], k_basis), 'spectral')][-1]:.4f} "
                  f"rand@{caps[-1]}={acc[(model_name, min(caps[-1], k_basis), 'random')][-1]:.4f}")

    # Save per-seed CSV
    if per_seed_rows:
        with open(out_csv_seed, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            w.writeheader()
            for r in per_seed_rows:
                w.writerow(r)
        print("\nSaved per-seed:", os.path.abspath(out_csv_seed))

    # Build mean/std rows
    mean_rows = []
    for (model_name, k_train, method), vals in sorted(acc.items(), key=lambda z: (z[0][0], z[0][1], z[0][2])):
        a = np.array(vals, dtype=float)
        mean_rows.append({
            "model": model_name,
            "k_train": int(k_train),
            "method": method,
            "mean_acc": float(a.mean()),
            "std_acc": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n_seeds": int(len(a)),
        })

    # Add NONE (mean/std) per model (as method="none", k_train="NA")
    for model_name, vals in none_acc.items():
        a = np.array(vals, dtype=float)
        mean_rows.append({
            "model": model_name,
            "k_train": "NA",
            "method": "none",
            "mean_acc": float(a.mean()),
            "std_acc": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n_seeds": int(len(a)),
        })

    # Save mean/std CSV
    if mean_rows:
        with open(out_csv_mean, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mean_rows[0].keys()))
            w.writeheader()
            for r in mean_rows:
                w.writerow(r)
        print("Saved mean/std:", os.path.abspath(out_csv_mean))

    # Plot mean ± std
    if PLOT and mean_rows:
        plt.figure(figsize=(10, 6))

        for model_name in sorted(set(r["model"] for r in mean_rows)):
            rows_m = [r for r in mean_rows if r["model"] == model_name]

            # NONE band
            none_row = [r for r in rows_m if r["method"] == "none"]
            if none_row:
                mu_none = none_row[0]["mean_acc"]
                sd_none = none_row[0]["std_acc"]
            else:
                mu_none, sd_none = None, None

            for method, marker, ls in [("spectral", "o", "-"), ("random", "x", "--")]:
                rows_k = [r for r in rows_m if r["method"] == method and r["k_train"] != "NA"]
                rows_k = sorted(rows_k, key=lambda z: int(z["k_train"]))
                ks = np.array([int(r["k_train"]) for r in rows_k], dtype=int)
                mu = np.array([r["mean_acc"] for r in rows_k], dtype=float)
                sd = np.array([r["std_acc"] for r in rows_k], dtype=float)

                plt.plot(ks, mu, marker=marker, linestyle=ls, label=f"{model_name} {method}")
                plt.fill_between(ks, mu - sd, mu + sd, alpha=0.2)

            # draw none horizontal line if available (over ks range)
            rows_k_any = [r for r in rows_m if r["method"] == "spectral" and r["k_train"] != "NA"]
            if mu_none is not None and rows_k_any:
                ks_any = sorted(int(r["k_train"]) for r in rows_k_any)
                xmin, xmax = min(ks_any), max(ks_any)
                plt.hlines(mu_none, xmin=xmin, xmax=xmax, linestyles=":", label=f"{model_name} none")
                if sd_none is not None and sd_none > 0:
                    plt.fill_between([xmin, xmax], [mu_none - sd_none, mu_none - sd_none],
                                     [mu_none + sd_none, mu_none + sd_none], alpha=0.15)

        plt.xlabel("k_train (projection rank used for training)")
        plt.ylabel("Target test accuracy (mean ± 1 std over seeds)")
        plt.title(f"Cap sweep: {args.src} → {args.tgt} (Ns={args.Ns}, Nt={args.Nt})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print("Saved plot:", os.path.abspath(out_png))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
