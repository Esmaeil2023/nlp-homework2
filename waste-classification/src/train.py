import os
import time
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report

# Import our own modules
from datasets import get_trashnet_loaders, CLASSES
from models import get_model, count_parameters

# ── What this file does ──────────────────────────────────────────────────────
# This is the main training loop. It:
#   1. Loads the dataset using datasets.py
#   2. Builds the model using models.py
#   3. Trains for N epochs, evaluating on validation set each epoch
#   4. Saves the best model checkpoint (lowest val loss)
#   5. Logs everything to Weights & Biases (wandb) for experiment tracking
#
# Key concepts:
#   - Epoch: one full pass through the training data
#   - Batch: a small group of images processed together (e.g. 32 images)
#   - Loss: how wrong the model is — we minimize this via backpropagation
#   - Accuracy: % of correct predictions
#   - Balanced Accuracy: accuracy that accounts for class imbalance
#     (important because TrashNet has unequal class sizes)
# ─────────────────────────────────────────────────────────────────────────────


# ── Configuration ─────────────────────────────────────────────────────────────
# All hyperparameters in one place — easy to change and track
CONFIG = {
    'model_name':   'resnet50',   # which model to train: resnet50, efficientnet, vit
    'data_root':    os.path.expanduser('~/waste-classification/data/raw/trashnet'),
    'num_classes':  6,
    'batch_size':   16,
    'num_epochs':   5,           # start with 5 epochs for the baseline
    'learning_rate': 1e-4,        # Adam learning rate
    'weight_decay': 1e-2,         # L2 regularization to prevent overfitting
    'train_ratio':  0.7,          # 70% of data for training
    'val_ratio':    0.15,         # 15% for validation, 15% for test
    'save_dir':     os.path.expanduser('~/waste-classification/experiments'),
    'use_wandb':    False,        # set to True after running: wandb login
    'seed':         42,
}
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    """
    Sets random seeds for reproducibility.
    With the same seed, you get the same results every run.
    Important for comparing experiments fairly.
    """
    torch.manual_seed(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """
    Returns the best available device.
    - MPS: Apple Silicon GPU (M1/M2/M3 Macs) — fastest on your machine
    - CUDA: NVIDIA GPU — fastest in general
    - CPU: fallback — slower but always available (your case: Intel Mac)
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    """
    Runs one full training epoch.

    For each batch:
      1. Move data to device (CPU/GPU)
      2. Forward pass: model predicts class probabilities
      3. Compute loss: how wrong were the predictions?
      4. Backward pass: compute gradients (how to adjust each weight)
      5. Optimizer step: update weights in the direction that reduces loss

    Returns a dict with average loss and balanced accuracy for this epoch.
    """
    model.train()  # puts model in training mode (enables dropout, batch norm updates)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to the compute device
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients from previous batch
        # (PyTorch accumulates gradients by default, so we reset each batch)
        optimizer.zero_grad()

        # Forward pass: get predictions
        outputs = model(images)  # shape: [batch_size, num_classes]

        # Compute loss
        # CrossEntropyLoss combines softmax + negative log likelihood
        # It penalizes the model more when it's confidently wrong
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)  # predicted class = highest score
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return {'loss': avg_loss, 'balanced_acc': bal_acc}


@torch.no_grad()  # disables gradient computation — saves memory and speeds up evaluation
def evaluate(model, loader, criterion, device) -> dict:
    """
    Evaluates the model on a validation or test set.

    No gradients are computed here — we're just measuring performance,
    not updating weights.

    Returns loss, balanced accuracy, and per-class metrics.
    """
    model.eval()  # puts model in eval mode (disables dropout, freezes batch norm)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Detailed per-class report (precision, recall, F1 per class)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    return {
        'loss': avg_loss,
        'balanced_acc': bal_acc,
        'report': report
    }


def train(config: dict):
    """
    Main training function.
    Trains the model for config['num_epochs'] epochs and saves the best checkpoint.
    """
    set_seed(config['seed'])
    device = get_device()
    print(f"\nUsing device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_trashnet_loaders(
        root=config['data_root'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding model: {config['model_name']}")
    model = get_model(config['model_name'], num_classes=config['num_classes'])
    model = model.to(device)  # move model to CPU/GPU

    params = count_parameters(model)
    print(f"  Trainable parameters: {params['trainable']:,}")

    # ── Loss function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss is standard for multi-class classification
    # We could add class weights here to handle imbalance (done later)
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW = Adam optimizer with decoupled weight decay
    # Better than SGD for fine-tuning pretrained models
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # ── Learning rate scheduler ───────────────────────────────────────────────
    # Cosine annealing: smoothly reduces LR from initial value to 0
    # This helps the model converge to a better minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )

    # ── Wandb setup ───────────────────────────────────────────────────────────
    if config['use_wandb']:
        wandb.init(
            project='waste-classification',
            name=f"{config['model_name']}-baseline",
            config=config
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(config['save_dir'], exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 60)

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{config['num_epochs']}")

        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Bal.Acc: {train_metrics['balanced_acc']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
              f"Val   Bal.Acc: {val_metrics['balanced_acc']:.4f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Log to wandb if enabled
        if config['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/balanced_acc': train_metrics['balanced_acc'],
                'val/loss': val_metrics['loss'],
                'val/balanced_acc': val_metrics['balanced_acc'],
                'lr': current_lr,
            })

        # Save best model checkpoint
        # We save when validation loss improves (model is generalizing better)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            checkpoint_path = os.path.join(
                config['save_dir'],
                f"{config['model_name']}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_balanced_acc': val_metrics['balanced_acc'],
                'config': config,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (epoch {epoch})")

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training complete! Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")
    print("\nEvaluating best model on test set...")

    # Load the best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Balanced Accuracy: {test_metrics['balanced_acc']:.4f}")
    print("\nPer-class results:")
    for cls in CLASSES:
        r = test_metrics['report'][cls]
        print(f"  {cls:12s} — precision: {r['precision']:.3f} | "
              f"recall: {r['recall']:.3f} | f1: {r['f1-score']:.3f}")

    if config['use_wandb']:
        wandb.log({'test/balanced_acc': test_metrics['balanced_acc']})
        wandb.finish()

    return test_metrics


if __name__ == '__main__':
    train(CONFIG)