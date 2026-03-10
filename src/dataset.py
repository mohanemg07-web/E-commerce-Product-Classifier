"""
Smart E-commerce Product Classifier — Dataset Preparation
==========================================================
Handles image loading, augmentation, train/val/test splitting,
and DataLoader creation for the product classification pipeline.
"""

import os
import sys
import json
import platform
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


# ── ImageNet normalization statistics ────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Default hyper-parameters ─────────────────────────────────
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
NUM_WORKERS    = 0 if platform.system() == "Windows" else 4
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15


def get_transforms(phase: str) -> transforms.Compose:
    """
    Return the appropriate image transforms for a given phase.

    Args:
        phase: One of 'train', 'val', or 'test'.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def prepare_datasets(
    data_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[Subset, Subset, Subset, list]:
    """
    Load images from *data_dir* (expected ImageFolder layout) and
    perform a reproducible 70 / 15 / 15 split.

    Returns:
        (train_dataset, val_dataset, test_dataset, class_names)
    """
    # Load full dataset with basic transforms to get total length
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = full_dataset.classes

    total        = len(full_dataset)
    train_size   = int(total * train_ratio)
    val_size     = int(total * val_ratio)
    test_size    = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply phase-specific transforms via wrapper datasets
    train_subset.dataset.transform = get_transforms("train")
    # We'll use separate copies for val/test transforms in the loaders
    return train_subset, val_subset, test_subset, class_names


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a ``Subset`` to override the parent dataset's transform."""

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # If PIL Image, apply transform
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        return image, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Prepare train, validation, and test DataLoaders.

    Args:
        data_dir:    Root directory with one sub-folder per class.
        batch_size:  Mini-batch size.
        num_workers: Parallel data-loading workers.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    # Build base dataset WITHOUT transforms so subsets get raw PIL images
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = full_dataset.classes

    total      = len(full_dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size   = int(total * VAL_RATIO)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_sub, val_sub, test_sub = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap each subset with the correct transform
    train_ds = _TransformSubset(train_sub, get_transforms("train"))
    val_ds   = _TransformSubset(val_sub,   get_transforms("val"))
    test_ds  = _TransformSubset(test_sub,  get_transforms("test"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Persist class mapping for downstream modules
    mapping_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "class_names.json",
    )
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"✅ Dataset loaded from: {data_dir}")
    print(f"   Classes ({len(class_names)}): {class_names}")
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_path)
    images, labels = next(iter(train_loader))
    print(f"   Batch shape : {images.shape}")
    print(f"   Label sample: {labels[:8].tolist()}")
