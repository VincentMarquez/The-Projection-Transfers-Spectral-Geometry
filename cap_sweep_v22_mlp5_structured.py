#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Transfers Cap Sweep + MLP Adapter + Spectral Scaling + Noise Baseline (5 seeds)
=====================================================================================

Improvements imported from Spectral Memory (structure-then-learn):

(1) Spectral coordinate scaling using eigenvalues:
    - sqrt:   z_i <- sqrt(lambda_i) * (u_i^T x)
    - whiten: z_i <- (u_i^T x) / sqrt(lambda_i + eps)
    Default: whiten (more stable for small-data MLP probes)

(2) Gaussian noise baseline:
    - Same MLP head architecture, trained on noise features of matching dimension.
    - Direct analogue of "replace CKL with noise" ablation logic.

(3) RevIN-style feature standardization (train-stat normalize):
    - Compute mean/std on TARGET TRAIN features (pre-projection)
    - Apply same normalization to train/val/test
    - Applied identically for NONE / RANDOM / SPECTRAL; removes scale confounds

Experiment:
- Hard pair (default): vehicles_all -> animals_all (target is 6-way)
- Caps: 32,64,99,kstar (kstar read from v22 CSV per backbone)
- MLP head: LayerNorm -> Linear -> GELU -> Dropout -> Linear
- Early stopping on VAL LOSS (stable vs val accuracy quantization)
- Stratified sampling for source/train/test and stratified train/val split
- 5 seeds mean ± std output and plot

Outputs:
  outputs_review/cap_sweep_struct_per_seed.csv
  outputs_review/cap_sweep_struct_mean_std.csv
  outputs_review/cap_sweep_struct_plot.png

Run (recommended):
  python3 cap_sweep_v22_mlp5_structured.py \
    --v22_csv scaling_results_multibackbone_v22.csv \
    --src vehicles_all --tgt animals_all \
    --Ns 2000 --Nt 100 --Ntest 2000 \
    --caps 32,64,99,kstar \
    --seeds 20260119,20260218,20260301,20260315,20260329 \
    --spectral_scale whiten \
    --epochs 250 --patience 35 --val_frac 0.30 \
    --hidden 256 --dropout 0.10 --lr 3e-3 --wd 1e-4

Author: Vincent Marquez
Date: Jan 2026
"""

import os
import csv
import argparse
import hashlib
import warnings
from collections import defaultdict
from typing import Tuple, List, Dict

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
# Deterministic hashing / seeding
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


# ----------------------------
# Stratified sampling + stratified split
# ----------------------------
def stratified_indices(labels_np: np.ndarray, class_list: List[int], total: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    classes_sorted = sorted(list(class_list))
    C = len(classes_sorted)
    per = total // C
    rem = total - per * C

    idx_all = []
    for j, c in enumerate(classes_sorted):
        want = per + (1 if j < rem else 0)
        idx_c = np.where(labels_np == c)[0]
        rng.shuffle(idx_c)
        take = min(want, len(idx_c))
        idx_all.extend(idx_c[:take].tolist())

    rng.shuffle(idx_all)
    return idx_all

def stratified_split(y: torch.Tensor, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    classes = np.unique(y_np)

    tr_idx, val_idx = [], []
    for c in classes:
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(val_frac * len(idx_c))))
        val_idx.extend(idx_c[:n_val].tolist())
        tr_idx.extend(idx_c[n_val:].tolist())

    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return np.array(tr_idx, dtype=int), np.array(val_idx, dtype=int)


# ----------------------------
# Feature extraction
# ----------------------------
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
# RevIN-style standardization (train-stat)
# ----------------------------
def fit_standardizer(X: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute featurewise mean/std on CPU for stability.
    X: [N, D]
    Returns (mu, sigma) as float32 tensors on CPU.
    """
    Xf = X.to(torch.float32)
    mu = Xf.mean(dim=0, keepdim=True)
    var = Xf.var(dim=0, unbiased=False, keepdim=True)
    sigma = torch.sqrt(var + eps)
    return mu, sigma

def apply_standardizer(X: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    Xf = X.to(torch.float32)
    return (Xf - mu) / sigma


# ----------------------------
# Covariance + KL basis (+ eigenvalues)
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
def kl_basis_eigh_with_lams(X_src: torch.Tensor, k: int, shrinkage: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      U_k:   [D, k] float32
      lams:  [k]    float32 (top-k eigenvalues, descending, clamped >=0)
    """
    Sigma = isotropic_shrunk_cov(X_src, shrinkage=shrinkage)  # float64
    evals, evecs = torch.linalg.eigh(Sigma)                   # ascending
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    k = int(min(k, evecs.shape[1]))
    lams = torch.clamp(evals[:k], min=0.0).to(torch.float32)
    U_k = evecs[:, :k].contiguous().to(torch.float32)
    return U_k, lams

@torch.no_grad()
def random_orthonormal_basis(D: int, k: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = torch.from_numpy(rng.standard_normal((D, k))).to(torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q.contiguous()


# ----------------------------
# CSV helper
# ----------------------------
def load_kstar_for_pair(v22_csv: str, model_name: str, src: str, tgt: str, k_field="k_star_median") -> int:
    with open(v22_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("model", "").strip() == model_name and
                r.get("source", "").strip() == src and
                r.get("target", "").strip() == tgt):
                return int(round(float(r[k_field])))
    raise KeyError(f"Missing row for {model_name} {src}->{tgt} in {v22_csv}")


# ----------------------------
# Learnable MLP adapter (Spectral-Memory style f_theta analogue)
# ----------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_with_early_stop(X_tr, y_tr, X_val, y_val,
                             in_dim, hidden, n_classes,
                             device, lr, wd, dropout,
                             epochs, patience, batch_size):
    """
    Early stop on VAL LOSS (stable for small val; less quantized than accuracy).
    """
    model = MLPHead(in_dim, hidden, n_classes, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_val = X_val.to(device); y_val = y_val.to(device)

    best_loss = float("inf")
    best_state = None
    bad = 0

    n = X_tr.shape[0]
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = F.cross_entropy(val_logits, y_val).item()

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss

@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    X = X.to(device); y = y.to(device)
    pred = model(X).argmax(dim=1)
    return float((pred == y).float().mean().item())


# ----------------------------
# Spectral scaling
# ----------------------------
def apply_spectral_scaling(Z: torch.Tensor, lams: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """
    Z:    [N, k]
    lams: [k]
    mode: "none" | "sqrt" | "whiten"
    """
    if mode == "none":
        return Z
    l = lams.to(Z.dtype).view(1, -1).clamp(min=0.0)
    if mode == "sqrt":
        return Z * torch.sqrt(l + eps)
    if mode == "whiten":
        return Z / torch.sqrt(l + eps)
    raise ValueError(f"Unknown spectral_scale mode: {mode}")


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

    ap.add_argument("--Ns", type=int, default=2000)
    ap.add_argument("--Nt", type=int, default=100)
    ap.add_argument("--Ntest", type=int, default=2000)

    ap.add_argument("--caps", type=str, default="32,64,99,kstar")
    ap.add_argument("--shrinkage", type=float, default=0.1)

    ap.add_argument("--seeds", type=str, default="20260119,20260218,20260301,20260315,20260329")
    ap.add_argument("--out_dir", type=str, default="outputs_review")

    # MLP hyperparams (adapter f_theta)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.30)

    # spectral scaling
    ap.add_argument("--spectral_scale", type=str, default="whiten", choices=["none","sqrt","whiten"])
    # noise baseline
    ap.add_argument("--use_noise", action="store_true", help="Include Gaussian noise baseline")
    ap.add_argument("--noise_seed_offset", type=int, default=777777)

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
    out_seed = os.path.join(args.out_dir, "cap_sweep_struct_per_seed.csv")
    out_mean = os.path.join(args.out_dir, "cap_sweep_struct_mean_std.csv")
    out_png = os.path.join(args.out_dir, "cap_sweep_struct_plot.png")

    tfm = imagenet_like_transform()
    ds_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    ds_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    y_train_np = np.array(ds_train.targets, dtype=np.int64)
    y_test_np  = np.array(ds_test.targets, dtype=np.int64)

    # Target label remap (within-target)
    tgt_classes_sorted = sorted(set(DOMAINS[tgt]))
    remap = {c:i for i,c in enumerate(tgt_classes_sorted)}
    C = len(tgt_classes_sorted)

    def remap_labels(y_raw: torch.Tensor) -> torch.Tensor:
        return torch.tensor([remap[int(v)] for v in y_raw.tolist()], dtype=torch.long)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    cap_tokens = [t.strip().lower() for t in args.caps.split(",") if t.strip()]

    per_seed_rows = []
    acc = defaultdict(list)       # (model, k_train, method) -> [acc]
    none_acc = defaultdict(list)  # model -> [acc]

    # include noise baseline aggregator if enabled
    noise_acc = defaultdict(list)

    for model_name in model_list:
        backbone, D = build_backbone(model_name, device)

        k_star = load_kstar_for_pair(args.v22_csv, model_name, src, tgt, k_field="k_star_median")
        k_star = max(1, min(k_star, D))

        caps = []
        for token in cap_tokens:
            if token == "kstar":
                caps.append(k_star)
            else:
                caps.append(int(token))
        caps = sorted(set(max(1, min(int(c), D)) for c in caps))

        print("\n" + "="*70)
        print(f"{model_name}: D={D} | {src}->{tgt} | k*={k_star} | caps={caps}")
        print(f"Seeds: {seed_list}")
        print(f"Standardize: train-stat pre-projection | spectral_scale={args.spectral_scale} | noise={args.use_noise}")
        print(f"MLP: hidden={args.hidden} dropout={args.dropout} lr={args.lr} wd={args.wd} "
              f"epochs={args.epochs} patience={args.patience} val_frac={args.val_frac}")
        print("="*70)

        for seed in seed_list:
            set_seed(seed)

            # SOURCE basis set (stratified)
            seed_src = stable_hash_tuple((seed, model_name, "train", src, args.Ns, "STRAT"))
            idx_src = stratified_indices(y_train_np, DOMAINS[src], args.Ns, seed=int(seed_src))
            loader_src = DataLoader(Subset(ds_train, idx_src), batch_size=128, shuffle=False, num_workers=0)
            Xs, _ = extract_features(backbone, loader_src, device, limit=len(idx_src))

            # TARGET train (stratified)
            seed_ttr = stable_hash_tuple((seed, model_name, "train", tgt, args.Nt, "STRAT"))
            idx_ttr = stratified_indices(y_train_np, DOMAINS[tgt], args.Nt, seed=int(seed_ttr))
            loader_ttr = DataLoader(Subset(ds_train, idx_ttr), batch_size=128, shuffle=False, num_workers=0)
            Xt_tr, yt_tr_raw = extract_features(backbone, loader_ttr, device, limit=len(idx_ttr))
            yt_tr = remap_labels(yt_tr_raw)

            # TARGET test (stratified)
            seed_tte = stable_hash_tuple((seed, model_name, "test", tgt, args.Ntest, "STRAT"))
            idx_tte = stratified_indices(y_test_np, DOMAINS[tgt], args.Ntest, seed=int(seed_tte))
            if len(idx_tte) < args.Ntest:
                warnings.warn(f"seed={seed} {model_name} test:{tgt} only has {len(idx_tte)} samples (requested {args.Ntest}).")
            loader_tte = DataLoader(Subset(ds_test, idx_tte), batch_size=128, shuffle=False, num_workers=0)
            Xt_te, yt_te_raw = extract_features(backbone, loader_tte, device, limit=len(idx_tte))
            yt_te = remap_labels(yt_te_raw)

            # -----------------------------
            # Feature standardization (RevIN-style) on TARGET TRAIN features (pre-projection)
            # -----------------------------
            mu, sigma = fit_standardizer(Xt_tr, eps=1e-6)
            Xt_tr_n = apply_standardizer(Xt_tr, mu, sigma)
            Xt_te_n = apply_standardizer(Xt_te, mu, sigma)

            # Stratified train/val split on target train labels
            tr_idx, val_idx = stratified_split(
                yt_tr, val_frac=args.val_frac,
                seed=stable_hash_tuple((seed, model_name, src, tgt, "VAL", args.Nt))
            )
            X_fit = Xt_tr_n[tr_idx]; y_fit = yt_tr[tr_idx]
            X_val = Xt_tr_n[val_idx]; y_val = yt_tr[val_idx]

            # NONE baseline (MLP on standardized full D)
            none_model, _ = train_mlp_with_early_stop(
                X_fit, y_fit, X_val, y_val,
                in_dim=D, hidden=args.hidden, n_classes=C,
                device=device, lr=args.lr, wd=args.wd, dropout=args.dropout,
                epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
            )
            a_none = accuracy(none_model, Xt_te_n, yt_te, device)
            none_acc[model_name].append(a_none)

            # KL basis up to k_star (capped by Ns-1)
            max_k_basis = max(1, min(D, int(Xs.shape[0]) - 1))
            k_basis = min(k_star, max_k_basis)
            U_full, lams_full = kl_basis_eigh_with_lams(Xs, k=k_basis, shrinkage=args.shrinkage)  # U:[D,k], lams:[k]

            for k_cap in caps:
                k_train = min(k_cap, k_basis)
                if k_train < 1:
                    continue

                # RANDOM projection (on standardized features)
                U_rand = random_orthonormal_basis(D, k_train,
                                                  seed=stable_hash_tuple((seed, model_name, src, tgt, "rand", k_train)))
                with torch.no_grad():
                    Z_fit_r = (X_fit.to(device) @ U_rand.to(device)).cpu()
                    Z_val_r = (X_val.to(device) @ U_rand.to(device)).cpu()
                    Z_te_r  = (Xt_te_n.to(device) @ U_rand.to(device)).cpu()

                rand_model, _ = train_mlp_with_early_stop(
                    Z_fit_r, y_fit, Z_val_r, y_val,
                    in_dim=k_train, hidden=args.hidden, n_classes=C,
                    device=device, lr=args.lr, wd=args.wd, dropout=args.dropout,
                    epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
                )
                a_r = accuracy(rand_model, Z_te_r, yt_te, device)

                # SPECTRAL projection (truncate KL basis) + spectral scaling
                U_k = U_full[:, :k_train]
                l_k = lams_full[:k_train]
                with torch.no_grad():
                    Z_fit = (X_fit.to(device) @ U_k.to(device)).cpu()
                    Z_val = (X_val.to(device) @ U_k.to(device)).cpu()
                    Z_te  = (Xt_te_n.to(device) @ U_k.to(device)).cpu()

                # Apply spectral scaling (whiten/sqrt/none)
                Z_fit = apply_spectral_scaling(Z_fit, l_k, mode=args.spectral_scale)
                Z_val = apply_spectral_scaling(Z_val, l_k, mode=args.spectral_scale)
                Z_te  = apply_spectral_scaling(Z_te,  l_k, mode=args.spectral_scale)

                spec_model, _ = train_mlp_with_early_stop(
                    Z_fit, y_fit, Z_val, y_val,
                    in_dim=k_train, hidden=args.hidden, n_classes=C,
                    device=device, lr=args.lr, wd=args.wd, dropout=args.dropout,
                    epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
                )
                a_s = accuracy(spec_model, Z_te, yt_te, device)

                acc[(model_name, k_train, "random")].append(a_r)
                acc[(model_name, k_train, "spectral")].append(a_s)

                # NOISE baseline (optional): same dims as k_train
                a_n = None
                if args.use_noise:
                    noise_seed = args.noise_seed_offset + stable_hash_tuple((seed, model_name, src, tgt, "noise", k_train))
                    rng = np.random.default_rng(int(noise_seed))
                    # Generate noise features for train/val/test matching shapes
                    Z_fit_n = torch.from_numpy(rng.standard_normal(size=(X_fit.shape[0], k_train))).to(torch.float32)
                    Z_val_n = torch.from_numpy(rng.standard_normal(size=(X_val.shape[0], k_train))).to(torch.float32)
                    Z_te_n  = torch.from_numpy(rng.standard_normal(size=(Xt_te_n.shape[0], k_train))).to(torch.float32)

                    noise_model, _ = train_mlp_with_early_stop(
                        Z_fit_n, y_fit, Z_val_n, y_val,
                        in_dim=k_train, hidden=args.hidden, n_classes=C,
                        device=device, lr=args.lr, wd=args.wd, dropout=args.dropout,
                        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
                    )
                    a_n = accuracy(noise_model, Z_te_n, yt_te, device)
                    noise_acc[(model_name, k_train, "noise")].append(a_n)

                per_seed_rows.append({
                    "seed": seed,
                    "model": model_name,
                    "D": D,
                    "source": src,
                    "target": tgt,
                    "Ns": int(args.Ns),
                    "Nt": int(args.Nt),
                    "Ntest": int(len(idx_tte)),
                    "k_star_csv": int(k_star),
                    "k_basis_used": int(k_basis),
                    "k_train": int(k_train),

                    "acc_none": float(a_none),
                    "acc_random": float(a_r),
                    "acc_spectral": float(a_s),
                    "acc_noise": ("" if a_n is None else float(a_n)),

                    "gain_spec_minus_rand": float(a_s - a_r),
                    "gain_spec_minus_none": float(a_s - a_none),
                    "gain_spec_minus_noise": ("" if a_n is None else float(a_s - a_n)),

                    "spectral_scale": args.spectral_scale,
                    "hidden": int(args.hidden),
                    "dropout": float(args.dropout),
                    "lr": float(args.lr),
                    "wd": float(args.wd),
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
                    "val_frac": float(args.val_frac),
                })

            print(f"  seed={seed} none={a_none:.4f}")

    # Save per-seed CSV
    if per_seed_rows:
        out_seed = os.path.join(args.out_dir, "cap_sweep_struct_per_seed.csv")
        with open(out_seed, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            w.writeheader()
            for r in per_seed_rows:
                w.writerow(r)
        print("\nSaved per-seed:", os.path.abspath(out_seed))

    # Mean/std rows
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

    # none mean/std
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

    # noise mean/std (if enabled)
    if args.use_noise:
        for (model_name, k_train, _), vals in sorted(noise_acc.items(), key=lambda z: (z[0][0], z[0][1])):
            a = np.array(vals, dtype=float)
            mean_rows.append({
                "model": model_name,
                "k_train": int(k_train),
                "method": "noise",
                "mean_acc": float(a.mean()),
                "std_acc": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "n_seeds": int(len(a)),
            })

    if mean_rows:
        with open(out_mean, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mean_rows[0].keys()))
            w.writeheader()
            for r in mean_rows:
                w.writerow(r)
        print("Saved mean/std:", os.path.abspath(out_mean))

    # Plot mean ± std
    if PLOT and mean_rows:
        plt.figure(figsize=(10, 6))

        for model_name in sorted(set(r["model"] for r in mean_rows)):
            rows_m = [r for r in mean_rows if r["model"] == model_name]

            none_row = [r for r in rows_m if r["method"] == "none"]
            mu_none = none_row[0]["mean_acc"] if none_row else None
            sd_none = none_row[0]["std_acc"] if none_row else None

            def plot_method(method, marker, ls):
                rows_k = [r for r in rows_m if r["method"] == method and r["k_train"] != "NA"]
                rows_k = sorted(rows_k, key=lambda z: int(z["k_train"]))
                if not rows_k:
                    return
                ks = np.array([int(r["k_train"]) for r in rows_k], dtype=int)
                mu = np.array([r["mean_acc"] for r in rows_k], dtype=float)
                sd = np.array([r["std_acc"] for r in rows_k], dtype=float)
                plt.plot(ks, mu, marker=marker, linestyle=ls, label=f"{model_name} {method}")
                plt.fill_between(ks, mu - sd, mu + sd, alpha=0.2)

            plot_method("spectral", "o", "-")
            plot_method("random", "x", "--")
            if args.use_noise:
                plot_method("noise", "+", ":")

            # none horizontal line + band across ks range
            rows_any = [r for r in rows_m if r["method"] == "spectral" and r["k_train"] != "NA"]
            if mu_none is not None and rows_any:
                ks_any = sorted(int(r["k_train"]) for r in rows_any)
                xmin, xmax = min(ks_any), max(ks_any)
                plt.hlines(mu_none, xmin=xmin, xmax=xmax, linestyles=":", label=f"{model_name} none")
                if sd_none is not None and sd_none > 0:
                    plt.fill_between([xmin, xmax],
                                     [mu_none - sd_none, mu_none - sd_none],
                                     [mu_none + sd_none, mu_none + sd_none],
                                     alpha=0.12)

        plt.xlabel("k_train (projection rank used for training)")
        plt.ylabel("Target test accuracy (mean ± 1 std over seeds)")
        plt.title(f"Project Transfers + MLP + scaling/noise/std: {args.src} → {args.tgt} (Ns={args.Ns}, Nt={args.Nt})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print("Saved plot:", os.path.abspath(out_png))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
