#!/usr/bin/env python3
"""
SpikeFormer demo — MNIST-like classification on synthetic data.

Shows:
  1. Model construction
  2. Training loop with spike rate monitoring
  3. Final accuracy and per-layer spike rates (energy efficiency proxy)

Run:
    python demo_basic.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from spikeformer import SpikeFormer, SpikeRateTracker


# ---------------------------------------------------------------------------
# Synthetic dataset (mimics 2D image patches flattened)
# ---------------------------------------------------------------------------

def make_dataset(n_samples=2000, n_classes=10, input_dim=64, seed=42):
    rng = torch.Generator()
    rng.manual_seed(seed)
    # Each class gets its own mean in high-dim space — linearly separable
    class_means = torch.randn(n_classes, input_dim, generator=rng)
    labels = torch.randint(0, n_classes, (n_samples,), generator=rng)
    x = class_means[labels] + 0.3 * torch.randn(n_samples, input_dim, generator=rng)
    return TensorDataset(x, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    spike_rates_accum = {}

    with SpikeRateTracker(model) as tracker:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, rates = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            # Accumulate per-batch rates (simple last-batch for demo clarity)
            spike_rates_accum = rates

    return total_loss / total, correct / total, spike_rates_accum


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    INPUT_DIM = 64
    NUM_CLASSES = 10
    DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    TIMESTEPS = 8
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 1e-3

    # Data
    dataset = make_dataset(n_samples=2000, n_classes=NUM_CLASSES, input_dim=INPUT_DIM)
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = SpikeFormer(
        input_dim=INPUT_DIM,
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        timesteps=TIMESTEPS,
        tau_mem=20.0,
        threshold=1.0,
        dropout=0.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SpikeFormer | params={total_params:,} | timesteps={TIMESTEPS}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("-" * 55)

    best_val_acc = 0.0
    final_rates = {}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, rates = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_rates = rates

        print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:9.3%}  {val_loss:9.4f}  {val_acc:8.3%}")

    # ---------------------------------------------------------------------------
    # Spike rate report
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"\nSpike rates (fraction of neurons firing per timestep):")
    print(f"  Lower → fewer synaptic operations → more energy-efficient")
    print(f"  Neuromorphic hardware ops ∝ spike rate (vs always-on for ANN)")
    print(f"{'-' * 55}")
    if final_rates:
        for layer, rate in sorted(final_rates.items()):
            bar = "█" * int(rate * 40)
            print(f"  {layer:<28} {rate:.3f}  {bar}")
        mean_rate = sum(final_rates.values()) / len(final_rates)
        print(f"{'-' * 55}")
        print(f"  {'mean':28} {mean_rate:.3f}")
        print(f"\nEstimated energy reduction vs ANN: {1/max(mean_rate, 0.01):.1f}× fewer MACs")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
