import torch
import torch.nn as nn
import timm

# ── What this file does ──────────────────────────────────────────────────────
# Defines the neural network architectures we will experiment with.
# We use `timm` (PyTorch Image Models) which gives us access to hundreds of
# pretrained models with one line of code.
#
# For this project we implement:
#   1. ResNet-50   — classic CNN, strong baseline, easy to interpret
#   2. EfficientNet-B3 — better accuracy/size tradeoff than ResNet
#   3. ViT-Small   — Vision Transformer, modern attention-based architecture
#
# All models are loaded with ImageNet pretrained weights, then we replace
# the final classification head to output 6 classes (our waste categories).
# ─────────────────────────────────────────────────────────────────────────────

# The 6 waste categories we classify into
NUM_CLASSES = 6
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    ResNet-50: A deep residual network with 50 layers.

    Why ResNet?
    - Very well studied and reliable baseline
    - Residual connections (skip connections) solve the vanishing gradient problem
    - Pretrained on ImageNet (1.2M images, 1000 classes) — transfers well to our task
    - The final 'fc' layer is replaced with our 6-class classifier

    Architecture summary:
        Input (3×224×224)
        → Convolutional layers with residual blocks
        → Global Average Pooling
        → Fully Connected (2048 → num_classes)
    """
    model = timm.create_model(
        'resnet50',
        pretrained=pretrained,   # load ImageNet weights
        num_classes=num_classes  # timm replaces the head automatically
    )
    return model


def build_efficientnet(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B3: A compound-scaled CNN.

    Why EfficientNet?
    - Scales width, depth, and resolution together (more efficient than just going deeper)
    - Better accuracy than ResNet50 with fewer parameters
    - B3 variant is a good middle ground between speed and accuracy

    Architecture summary:
        Input (3×224×224)
        → MBConv blocks (mobile inverted bottleneck + squeeze-excitation)
        → Global Average Pooling
        → Dropout → Fully Connected (1536 → num_classes)
    """
    model = timm.create_model(
        'efficientnet_b3',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def build_vit(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Vision Transformer Small (ViT-S/16): Attention-based architecture.

    Why ViT?
    - Treats image as a sequence of 16×16 patches (like words in a sentence)
    - Uses self-attention to learn global relationships between patches
    - No inductive bias about locality — must learn spatial structure from data
    - Interesting for domain generalization research: different failure modes than CNNs

    Architecture summary:
        Input (3×224×224)
        → Split into 196 patches of size 16×16
        → Linear embedding + positional encoding
        → 12 Transformer encoder blocks (self-attention + MLP)
        → [CLS] token → Fully Connected (384 → num_classes)

    Note: ViT needs pretrained weights to work well — it requires lots of data
    to learn from scratch. With pretrained weights it transfers well.
    """
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def get_model(model_name: str, num_classes: int = NUM_CLASSES,
              pretrained: bool = True) -> nn.Module:
    """
    Factory function — returns a model by name string.
    This makes it easy to switch models from a config or command line argument.

    Args:
        model_name: one of 'resnet50', 'efficientnet', 'vit'
        num_classes: number of output classes (default 6 for TrashNet)
        pretrained: whether to load ImageNet pretrained weights

    Returns:
        nn.Module: the requested model, ready for training
    """
    builders = {
        'resnet50':     build_resnet50,
        'efficientnet': build_efficientnet,
        'vit':          build_vit,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(builders.keys())}"
        )

    model = builders[model_name](num_classes=num_classes, pretrained=pretrained)
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Counts total and trainable parameters in a model.
    Useful for comparing model sizes.

    A parameter is 'trainable' if it has requires_grad=True,
    meaning gradients will be computed for it during backpropagation.
    Frozen layers (requires_grad=False) won't be updated during training.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def freeze_backbone(model: nn.Module, unfreeze_last_n_layers: int = 0) -> nn.Module:
    """
    Freezes the backbone (feature extractor) of the model.
    Only the classification head will be trained.

    Why freeze?
    - Much faster training — only a small number of parameters are updated
    - Useful when your dataset is small (TrashNet has only ~2500 images)
    - The backbone already learned good features from ImageNet

    Args:
        model: the model to freeze
        unfreeze_last_n_layers: if > 0, also unfreeze the last N layers
                                of the backbone for fine-tuning

    Returns:
        model with frozen backbone
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the classification head (last layer)
    # timm models store the head differently depending on architecture
    if hasattr(model, 'fc'):          # ResNet style
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):  # EfficientNet style
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'head'):      # ViT style
        for param in model.head.parameters():
            param.requires_grad = True

    return model


if __name__ == '__main__':
    # Quick test — build each model and print parameter counts
    print("=" * 50)
    for name in ['resnet50', 'efficientnet', 'vit']:
        print(f"\nBuilding {name}...")
        model = get_model(name, pretrained=False)  # pretrained=False for speed
        params = count_parameters(model)
        print(f"  Total parameters:     {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")

        # Test a forward pass with a dummy batch
        # Batch of 2 images, 3 channels, 224x224 pixels
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")  # should be [2, 6]
        assert output.shape == (2, NUM_CLASSES), "Output shape mismatch!"
        print(f"  {name} OK ✓")

    print("\n" + "=" * 50)
    print("models.py working correctly!")