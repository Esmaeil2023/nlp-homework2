import os
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms


# Unified class mapping — all datasets will map to these 6 classes
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class TrashNetDataset(Dataset):
    """
    Loads the TrashNet dataset from a folder with structure:
        root/
            cardboard/
            glass/
            metal/
            paper/
            plastic/
            trash/
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # list of (image_path, label_index)

        for class_name in CLASSES:
            class_dir = self.root / class_name
            if not class_dir.exists():
                print(f"Warning: class folder not found: {class_dir}")
                continue
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), CLASS_TO_IDX[class_name]))

        print(f"TrashNetDataset loaded: {len(self.samples)} images from {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split: str = 'train'):
    """Returns image transforms for train or val/test splits."""
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_trashnet_loaders(root: str, batch_size: int = 32,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15):
    train_dataset = TrashNetDataset(root, transform=get_transforms('train'))
    val_dataset   = TrashNetDataset(root, transform=get_transforms('val'))
    test_dataset  = TrashNetDataset(root, transform=get_transforms('val'))

    n = len(train_dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(train_dataset, train_idx),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(val_dataset, val_idx),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(Subset(test_dataset, test_idx),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Split — train: {n_train}, val: {n_val}, test: {n_test}")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    DATA_ROOT = os.path.expanduser('~/waste-classification/data/raw/trashnet')

    dataset = TrashNetDataset(DATA_ROOT, transform=get_transforms('val'))
    print(f"Total images: {len(dataset)}")

    img, label = dataset[0]
    print(f"Label: {label} = {CLASSES[label]}")
    print(f"Type: {type(img)}")
    if hasattr(img, 'shape'):
        print(f"Shape: {img.shape}")

    print("datasets.py working correctly!")