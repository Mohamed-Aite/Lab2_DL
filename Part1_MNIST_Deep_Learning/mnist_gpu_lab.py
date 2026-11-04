"""GPU-ready MNIST laboratory: CNN, Faster R-CNN, transfer learning and ViT.

The script is intentionally configurable for Colab/Kaggle.  It downloads MNIST
through torchvision, never stores datasets or checkpoints in the repository,
and reports accuracy, macro-F1, loss and elapsed training time.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    seconds: float


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loaders(data_dir: str, batch_size: int, limit: int | None = None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    if limit:
        train = Subset(train, range(min(limit, len(train))))
        test = Subset(test, range(min(max(1000, limit // 5), len(test))))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def transfer_loaders(data_dir: str, batch_size: int, limit: int | None = None):
    """Create ImageNet-compatible 3-channel 224x224 MNIST batches."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    if limit:
        train = Subset(train, range(min(limit, len(train))))
        test = Subset(test, range(min(max(1000, limit // 5), len(test))))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128),
                                        nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 10))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class FCNN(nn.Module):
    """Fully connected baseline for comparison with spatial models."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, dim=128, depth=4, heads=4):
        super().__init__()
        patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, dim, patch_size, patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, patches + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 2,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        return self.head(self.encoder(x)[:, 0])


def scores(pred: Tensor, target: Tensor) -> Tuple[int, int, List[int], List[int]]:
    p, t = pred.detach().cpu().tolist(), target.detach().cpu().tolist()
    return len(p), sum(a == b for a, b in zip(p, t)), p, t


@torch.no_grad()
def evaluate(model: nn.Module, data: Iterable, criterion: nn.Module, started: float) -> Metrics:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    cm = torch.zeros(10, 10, dtype=torch.long)
    for x, y in data:
        out = model(x.to(next(model.parameters()).device))
        loss_sum += criterion(out, y.to(out.device)).item() * len(y)
        p = out.argmax(1).cpu()
        total += len(y); correct += int((p == y).sum())
        for actual, predicted in zip(y, p):
            cm[actual, predicted] += 1
    f1 = []
    for c in range(10):
        tp = cm[c, c].float()
        precision = tp / cm[:, c].sum().clamp_min(1)
        recall = tp / cm[c, :].sum().clamp_min(1)
        f1.append((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())
    return Metrics(loss_sum / total, correct / total, sum(f1) / 10, time.perf_counter() - started)


def train(model: nn.Module, train_data: DataLoader, test_data: DataLoader, epochs: int, lr: float) -> Metrics:
    model.to(device())
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in train_data:
            x, y = x.to(device(), non_blocking=True), y.to(device(), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return evaluate(model, test_data, criterion, started)


def transfer_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
    elif name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
    else:
        raise ValueError(f"Unsupported transfer model: {name}")
    model = getattr(models, name)(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 10)
    return model


def faster_rcnn_for_mnist() -> nn.Module:
    """Build a detector for a synthetic MNIST box protocol.

    MNIST itself has no boxes.  A training adapter must create one foreground
    box per image and use labels 1..10 (0 is the detector background class).
    The adapter is deliberately separate so users do not mistake image-level
    MNIST labels for real detection annotations.
    """
    return fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=11,
        min_size=28, max_size=28,
    )


def mnist_detection_target(image: Tensor, digit: int) -> Dict[str, Tensor]:
    """Return a synthetic full-image target for demonstration only."""
    _, height, width = image.shape
    return {
        "boxes": torch.tensor([[0.0, 0.0, float(width), float(height)]]),
        "labels": torch.tensor([digit + 1], dtype=torch.int64),
        "image_id": torch.tensor([digit], dtype=torch.int64),
        "area": torch.tensor([float(width * height)]),
        "iscrowd": torch.zeros(1, dtype=torch.int64),
    }


def detection_batch(batch):
    images, targets = zip(*batch)
    images = [image.repeat(3, 1, 1) for image in images]
    return list(images), list(targets)


def detection_loader(data_dir: str, batch_size: int, limit: int | None = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))

    class DetectionDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            image, digit = dataset[index]
            return image, mnist_detection_target(image, int(digit))

    return DataLoader(DetectionDataset(), batch_size=batch_size, shuffle=True,
                      collate_fn=detection_batch, num_workers=2, pin_memory=True)


def train_faster_rcnn(model: nn.Module, data: DataLoader, epochs: int = 1,
                      lr: float = 1e-4) -> Metrics:
    """Train/evaluate on synthetic full-image boxes for an explicit protocol.

    This is a demonstration adapter, not a claim that original MNIST contains
    object annotations. Its accuracy and macro-F1 measure digit-label recovery
    from the predicted box, while loss is the detector training loss.
    """
    model.to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    started = time.perf_counter()
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        for images, targets in data:
            images = [image.to(device(), non_blocking=True) for image in images]
            targets = [{key: value.to(device()) for key, value in target.items()}
                       for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            last_loss = float(loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    with torch.no_grad():
        for images, targets in data:
            predictions = model([image.to(device()) for image in images])
            for prediction, target in zip(predictions, targets):
                predicted = int(prediction["labels"][0].cpu()) - 1 if len(prediction["labels"]) else -1
                actual = int(target["labels"][0]) - 1
                if 0 <= predicted < 10:
                    confusion[actual, predicted] += 1
                correct += int(predicted == actual)
                total += 1
    f1 = []
    for cls in range(10):
        tp = confusion[cls, cls].float()
        precision = tp / confusion[:, cls].sum().clamp_min(1)
        recall = tp / confusion[cls, :].sum().clamp_min(1)
        f1.append((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())
    return Metrics(last_loss, correct / max(total, 1), sum(f1) / 10,
                   time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train_data, test_data = loaders(args.data_dir, args.batch_size, args.limit)
    transfer_train, transfer_test = transfer_loaders(args.data_dir, args.batch_size, args.limit)
    experiments = {"FCNN": FCNN(), "CNN": CNN(), "ViT": ViT()}
    for name, model in experiments.items():
        print(name, train(model, train_data, test_data, args.epochs, 1e-3))
    for name in ("vgg16", "alexnet"):
        print(name, train(transfer_model(name, pretrained=True), transfer_train, transfer_test, args.epochs, 1e-4))
    detector_data = detection_loader(args.data_dir, max(8, args.batch_size // 8), args.limit)
    print("Faster R-CNN:", train_faster_rcnn(
        faster_rcnn_for_mnist(), detector_data, max(1, args.epochs // 2)))


if __name__ == "__main__":
    main()
