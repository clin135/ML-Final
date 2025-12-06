import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- SAS library imports ---
from sas.approx_latent_classes import kmeans_approx
from sas.subset_dataset import SASSubsetDataset, RandomSubsetDataset

# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_simclr_transform():
    # SimCLR-style augmentations for CIFAR-10
    return transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
            std=(0.2470, 0.2435, 0.2616),    # CIFAR-10 std
        ),
    ])


def get_eval_transform():
    # Eval transform for CIFAR-10
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])


# --------------------------------------------------------------------------------
# Dataset wrapper that produces two views
# --------------------------------------------------------------------------------

class SimCLRPairDataset(Dataset):
    """
    Wraps a base dataset that returns (image, label, ...) and produces two
    augmented views of the same underlying image for SimCLR.
    Handles both PIL images and tensor images.
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, (list, tuple)):
            img = item[0]
        else:
            img = item

        # If img is a tensor (e.g., from ToTensor), convert back to PIL
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


# --------------------------------------------------------------------------------
# SimCLR model + loss
# --------------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        feat_dim = 512
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feature_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.proj_head(h)
        return h, z


def simclr_nt_xent_loss(z, temperature=0.5):
    """
    z: (2N, d) projection vectors; first N are view1, next N are view2.
    """
    device = z.device
    z = F.normalize(z, dim=1)
    N = z.size(0) // 2

    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)
    mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -1e9)

    pos_indices = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(0, N)
    ]).to(device)

    loss = F.cross_entropy(sim, pos_indices)
    return loss


# --------------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------------

def pretrain_simclr(dataset, device, epochs=10, batch_size=256,
                    lr=1e-3, temperature=0.5, proj_dim=128):
    """
    Pretrain a SimCLR encoder on the given dataset (which should yield (x1, x2)).
    Returns the trained encoder (without projection head).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    model = SimCLRModel(feature_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[INFO] SimCLR pretraining for {epochs} epochs on {len(dataset)} samples")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        for x1, x2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            z = torch.cat([z1, z2], dim=0)
            loss = simclr_nt_xent_loss(z, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"[SimCLR] Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    return model.encoder  # only backbone


class LinearEvalModel(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
            h = h.view(h.size(0), -1)
        logits = self.fc(h)
        return logits


def linear_evaluation(encoder, device, epochs=20, batch_size=256):
    """
    Standard linear eval on frozen encoder using full CIFAR10 train set.
    Returns best test accuracy (%) over epochs.
    """
    transform = get_eval_transform()
    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    model = LinearEvalModel(encoder, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.fc.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total * 100.0

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                _, pred = logits.max(1)
                correct += pred.eq(y).sum().item()
                total += x.size(0)

        test_acc = correct / total * 100.0
        best_acc = max(best_acc, test_acc)
        print(
            f"[Linear] Epoch {epoch}/{epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
            f"Test Acc {test_acc:.2f}% (Best {best_acc:.2f}%)"
        )

    return best_acc


# --------------------------------------------------------------------------------
# Build SAS / Random indices (not datasets)
# --------------------------------------------------------------------------------

def build_sas_and_random_indices(device, fractions) -> (Dict[float, List[int]], Dict[float, List[int]]):
    """
    Uses the sas library to compute SAS and random subset indices for CIFAR10.
    Returns:
        sas_indices[frac] = list[int]
        rand_indices[frac] = list[int]
    """
    sas_indices: Dict[float, List[int]] = {}
    rand_indices: Dict[float, List[int]] = {}

    # Base dataset for SAS (simple tensor transform)
    base_transform = transforms.ToTensor()
    cifar_for_sas = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=base_transform,
    )
    num_train = len(cifar_for_sas)

    # Load proxy model from repo (CIFAR-10 version)
    proxy_path = "proxy-cifar10-resnet10-399-net.pt"
    if not os.path.exists(proxy_path):
        raise FileNotFoundError(
            f"{proxy_path} not found. Make sure you're in the "
            "sjoshi804/sas-data-efficient-contrastive-learning repo."
        )

    print("[INFO] Loading proxy model for SAS latent classes...")
    try:
        proxy_model = torch.load(proxy_path, map_location=device, weights_only=False)
    except TypeError:
        proxy_model = torch.load(proxy_path, map_location=device)
    proxy_model.to(device)
    proxy_model.eval()

    # Approximate latent class partition once
    print("[INFO] Computing approximate latent classes with kmeans_approx...")
    try:
        partition = kmeans_approx(
            cifar_for_sas,
            proxy_model,
            num_classes=10,
            device=device,
            batch_size=512,
            num_workers=2,
            verbose=True,
        )
    except TypeError:
        print("[WARN] kmeans_approx doesn't support batch_size/num_workers/verbose kwargs; "
              "calling with a simpler signature.")
        partition = kmeans_approx(
            cifar_for_sas,
            proxy_model,
            10,
            device,
        )
    print("[INFO] Latent class partition computed.")

    for frac in fractions:
        print(f"[INFO] Building SAS & Random indices for frac={frac}...")
        # Build SASSubsetDataset JUST to get its subset_indices
        sas_ds = SASSubsetDataset(
            dataset=cifar_for_sas,
            subset_fraction=frac,
            num_downstream_classes=10,
            device=device,
            proxy_model=proxy_model,
            approx_latent_class_partition=partition,
            verbose=True,
        )
        # subset_indices should be a list / array of ints
        if hasattr(sas_ds, "subset_indices"):
            sas_idx = [int(i) for i in list(sas_ds.subset_indices)]
        else:
            raise RuntimeError("SASSubsetDataset has no attribute 'subset_indices'")
        sas_indices[frac] = sas_idx
        print(f"[INFO] SAS frac={frac}: got {len(sas_idx)} indices (train size={num_train})")

        # Build RandomSubsetDataset JUST to get subset_indices
        rand_ds = RandomSubsetDataset(
            dataset=cifar_for_sas,
            subset_fraction=frac,
        )
        if hasattr(rand_ds, "subset_indices"):
            rand_idx = [int(i) for i in list(rand_ds.subset_indices)]
        else:
            raise RuntimeError("RandomSubsetDataset has no attribute 'subset_indices'")
        rand_indices[frac] = rand_idx
        print(f"[INFO] Random frac={frac}: got {len(rand_idx)} indices (train size={num_train})")

    return sas_indices, rand_indices


# --------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------

def main():
    set_seed(0)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Fractions: 20%, 40%, 60%, 80%, 90%, 100%
    fractions = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    # Training settings
    pretrain_epochs = 50   # SimCLR epochs
    linear_epochs = 20     # linear eval epochs
    batch_size = 256

    # 1. Full data baseline
    print("\n========== FULL DATA ==========")
    cifar_full_raw = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=None,        # get PIL images
    )
    simclr_full = SimCLRPairDataset(cifar_full_raw, transform=get_simclr_transform())
    encoder_full = pretrain_simclr(
        simclr_full, device,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=1e-3,
        temperature=0.5,
        proj_dim=128,
    )
    full_acc = linear_evaluation(
        encoder_full, device,
        epochs=linear_epochs,
        batch_size=batch_size,
    )
    print(f"[RESULT] Full-data best test accuracy: {full_acc:.2f}%")

    # 2. Build SAS + random subset indices
    print("\n========== BUILDING SAS + RANDOM SUBSET INDICES ==========")
    sas_idx_dict, rand_idx_dict = build_sas_and_random_indices(device, fractions)

    # Base raw CIFAR10 (PIL) to build actual subsets for SimCLR
    base_raw = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=None,    # PIL
    )

    # 3. For each fraction: train SimCLR on SAS subset & random subset
    sas_accs = []
    rand_accs = []

    for frac in fractions:
        print(f"\n========== FRACTION = {frac * 100:.0f}% (SAS) ==========")
        sas_subset = Subset(base_raw, sas_idx_dict[frac])
        simclr_sas = SimCLRPairDataset(sas_subset, transform=get_simclr_transform())
        encoder_sas = pretrain_simclr(
            simclr_sas, device,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            lr=1e-3,
            temperature=0.5,
            proj_dim=128,
        )
        acc_sas = linear_evaluation(
            encoder_sas, device,
            epochs=linear_epochs,
            batch_size=batch_size,
        )
        sas_accs.append(acc_sas)
        print(f"[RESULT] SAS {frac*100:.0f}% best test accuracy: {acc_sas:.2f}%")

        print(f"\n========== FRACTION = {frac * 100:.0f}% (Random) ==========")
        rand_subset = Subset(base_raw, rand_idx_dict[frac])
        simclr_rand = SimCLRPairDataset(rand_subset, transform=get_simclr_transform())
        encoder_rand = pretrain_simclr(
            simclr_rand, device,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            lr=1e-3,
            temperature=0.5,
            proj_dim=128,
        )
        acc_rand = linear_evaluation(
            encoder_rand, device,
            epochs=linear_epochs,
            batch_size=batch_size,
        )
        rand_accs.append(acc_rand)
        print(f"[RESULT] Random {frac*100:.0f}% best test accuracy: {acc_rand:.2f}%")

    # 4. Plot like Figure (b) – CIFAR10 – show in notebook (no saving)
    x_percent = [f * 100 for f in fractions]

    plt.figure(figsize=(5, 4))
    plt.plot(x_percent, rand_accs, marker="o", label="Random")
    plt.plot(x_percent, sas_accs, marker="o", label="SAS")
    plt.axhline(full_acc, linestyle="--", label="Full Data")
    plt.xlabel("Subset Size (%)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("CIFAR10 SimCLR + SAS vs Random")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
