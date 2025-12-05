import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 512
EPOCHS = 100
NOISE_RATIO = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {DEVICE}")
print(f"Noise Level: {NOISE_RATIO*100}%")

class CorruptedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, noise_ratio=0.2):
        super().__init__(root, train=train, transform=transform, download=download)

        num_samples = len(self.data)
        num_noise = int(num_samples * noise_ratio)
        noise_indices = np.random.choice(num_samples, num_noise, replace=False)

        print(f"Injecting noise into {num_noise} images...")

        noise_data = np.random.randint(0, 256, size=(num_noise, 32, 32, 3), dtype='uint8')
        self.data[noise_indices] = noise_data
        self.noise_indices = set(noise_indices)

def visualize_dataset_noise(dataset, num_samples=5):
    all_indices = np.arange(len(dataset))
    noise_indices = list(dataset.noise_indices)
    real_indices = [i for i in all_indices if i not in dataset.noise_indices]

    sample_real = np.random.choice(real_indices, num_samples, replace=False)
    sample_noise = np.random.choice(noise_indices, num_samples, replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    for i, idx in enumerate(sample_real):
        axes[0, i].imshow(dataset.data[idx])
        axes[0, i].set_title(f"Real: {dataset.targets[idx]}")
        axes[0, i].axis('off')

    for i, idx in enumerate(sample_noise):
        axes[1, i].imshow(dataset.data[idx])
        axes[1, i].set_title(f"Noise (Label: {dataset.targets[idx]})")
        axes[1, i].axis('off')

    plt.suptitle(f"Real vs. {NOISE_RATIO*100}% Noise", fontsize=16)
    plt.tight_layout()
    plt.show()

class SimCLR(nn.Module):
    def __init__(self, base_model):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        self.feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size).to(z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        return self.criterion(sim_matrix, labels) / (2 * batch_size)

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, indices=None):
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else list(range(len(base_dataset)))
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, target = self.base_dataset[real_idx]
        return (self.transform(img), self.transform(img)), target

def get_sas_indices(dataset, keep_ratio=1.0):
    if keep_ratio == 1.0:
        return list(range(len(dataset)))

    proxy = torchvision.models.resnet18(pretrained=True).to(DEVICE)
    proxy.fc = nn.Identity()
    proxy.eval()

    t = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    class FeatureDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            img, label = self.ds[i]
            return t(img), label

    loader = DataLoader(FeatureDataset(dataset), batch_size=256, shuffle=False)

    feats, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting Features"):
            feats.append(proxy(x.to(DEVICE)).cpu())
            targets.append(y)

    feats = torch.cat(feats)
    targets = torch.cat(targets)

    keep_indices = []
    for c in range(10):
        mask = (targets == c)
        idxs = torch.nonzero(mask).squeeze()
        c_feats = feats[mask]

        centroid = c_feats.mean(0, keepdim=True)
        sims = F.cosine_similarity(c_feats, centroid)

        sorted_sims, sorted_idxs = torch.sort(sims, descending=True)
        num = int(len(idxs) * keep_ratio)
        keep_indices.extend(idxs[sorted_idxs[:num]].tolist())

    print(f"SAS kept {len(keep_indices)} images.")
    return keep_indices

def evaluate_linear(model, train_dataset, val_dataset, train_indices=None):
    model.eval()

    plain_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    class PlainDataset(Dataset):
        def __init__(self, base_ds, indices=None):
            self.base = base_ds
            self.indices = indices if indices is not None else list(range(len(base_ds)))
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, target = self.base[self.indices[idx]]
            img = plain_transform(img)
            return img, target

    train_plain = PlainDataset(train_dataset, indices=train_indices)
    val_plain = PlainDataset(val_dataset, indices=None)

    def extract_features(dataset):
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        feats, labels = [], []
        with torch.no_grad():
            for img, target in loader:
                img = img.to(DEVICE)
                h, _ = model(img)
                feats.append(h.cpu())
                labels.append(target)
        return torch.cat(feats), torch.cat(labels)

    train_X, train_y = extract_features(train_plain)
    val_X, val_y = extract_features(val_plain)

    linear = nn.Linear(train_X.size(1), 10).to(DEVICE)
    opt = optim.Adam(linear.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    linear.train()
    batch_size = 256
    for i in range(0, len(train_X), batch_size):
        x = train_X[i:i + batch_size].to(DEVICE)
        y = train_y[i:i + batch_size].to(DEVICE)

        opt.zero_grad()
        logits = linear(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    def accuracy(X, y):
        linear.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                logits = linear(X[i:i + 512].to(DEVICE))
                preds.append(torch.argmax(logits, dim=1).cpu())
        preds = torch.cat(preds)
        return (preds == y).float().mean().item()

    train_acc = accuracy(train_X, train_y)
    val_acc = accuracy(val_X, val_y)
    return train_acc, val_acc

def run_experiment(name, train_indices, train_dataset, val_dataset):
    print(f"\n--- Experiment: {name} ---")

    train_ds = SimCLRDataset(train_dataset, train_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_ds = SimCLRDataset(val_dataset)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    base_encoder = torchvision.models.resnet18(pretrained=False)
    base_encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    base_encoder.maxpool = nn.Identity()
    model = SimCLR(base_encoder).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            optimizer.zero_grad()
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x1, x2), _ in val_loader:
                x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = criterion(z1, z2)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        train_acc, val_acc = evaluate_linear(
            model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_indices=train_indices
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%"
        )

    return history


if __name__ == "__main__":
    # datasets
    train_dataset = CorruptedCIFAR10(
        root='./data',
        train=True,
        download=True,
        noise_ratio=NOISE_RATIO
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True
    )

    visualize_dataset_noise(train_dataset)

    indices_full = list(range(len(train_dataset)))
    indices_sas = get_sas_indices(train_dataset, keep_ratio=1.0 - NOISE_RATIO)

    hist_full = run_experiment("Baseline (40% Noise)", indices_full, train_dataset, val_dataset)
    hist_sas = run_experiment("SAS (Top 60% Only)", indices_sas, train_dataset, val_dataset)


    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist_full['train_loss'], label='Baseline Train Loss', linestyle='--')
    plt.plot(epochs_range, hist_full['val_loss'], label='Baseline Val Loss')
    plt.plot(epochs_range, hist_sas['train_loss'], label='SAS Train Loss', linestyle='--')
    plt.plot(epochs_range, hist_sas['val_loss'], label='SAS Val Loss')
    plt.title("Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in hist_full['train_acc']],
             label='Baseline Train Acc', linestyle='--')
    plt.plot(epochs_range, [a * 100 for a in hist_full['val_acc']],
             label='Baseline Val Acc')
    plt.plot(epochs_range, [a * 100 for a in hist_sas['train_acc']],
             label='SAS Train Acc', linestyle='--')
    plt.plot(epochs_range, [a * 100 for a in hist_sas['val_acc']],
             label='SAS Val Acc')
    plt.title("Linear Eval Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
