"""Experiment 4: AlexNet image classification on MNIST.

The script follows the course handout:
1. build an AlexNet-style CNN;
2. split MNIST into train/validation/test sets with a 7:2:1 ratio;
3. record training and validation losses/accuracies;
4. visualize convolution filters and intermediate feature maps;
5. run hyper-parameter experiments for learning rate, batch size and dropout.

Examples
--------
Full base experiment:
    python prog4/alexnet_mnist.py --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "datasets"
DEFAULT_RESULT_DIR = PROJECT_DIR / "results"
DEFAULT_LOG_DIR = PROJECT_DIR / "log"


@dataclass(frozen=True)
class Metrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class AlexNet(nn.Module):
    """AlexNet-style network adapted for one-channel MNIST inputs."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(
    num_classes: int,
    dropout: float,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
    model = AlexNet(num_classes=num_classes, dropout=dropout).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if verbose:
            print(f"Using {torch.cuda.device_count()} CUDA GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> AlexNet:
    if isinstance(model, nn.DataParallel):
        return model.module
    if isinstance(model, AlexNet):
        return model
    raise TypeError(f"Expected AlexNet or DataParallel[AlexNet], got {type(model)!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_dataset(data_dir: Path) -> torch.utils.data.Dataset:
    transform = build_transform()
    return datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )


def split_dataset(
    dataset: torch.utils.data.Dataset,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def write_metrics_csv(path: Path, rows: Iterable[Metrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(Metrics(0, 0, 0, 0, 0)).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    log_path: Path,
    result_save_path: Path,
    show_progress: bool = False,
    verbose: bool = True,
) -> list[Metrics]:
    result_save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics: list[Metrics] = []
    writer = SummaryWriter(log_dir=str(log_path))
    result_txt = result_save_path / "result.txt"

    with result_txt.open("w", encoding="utf-8") as f:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs} train",
                disable=not show_progress,
            )
            for images, labels in train_iter:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                val_iter = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch}/{epochs} val",
                    disable=not show_progress,
                )
                for images, labels in val_iter:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            row = Metrics(
                epoch=epoch,
                train_loss=train_loss / max(len(train_loader), 1),
                train_acc=100.0 * correct_train / max(total_train, 1),
                val_loss=val_loss / max(len(val_loader), 1),
                val_acc=100.0 * correct_val / max(total_val, 1),
            )
            metrics.append(row)
            writer.add_scalar("Train Loss", row.train_loss, epoch)
            writer.add_scalar("Train Acc", row.train_acc, epoch)
            writer.add_scalar("Val Loss", row.val_loss, epoch)
            writer.add_scalar("Val Acc", row.val_acc, epoch)

            line = (
                f"Epoch {epoch}/{epochs}, "
                f"Train Loss: {row.train_loss:.4f}, Train Acc: {row.train_acc:.2f}%, "
                f"Val Loss: {row.val_loss:.4f}, Val Acc: {row.val_acc:.2f}%"
            )
            if verbose:
                print(line)
            f.write(line + "\n")
            f.flush()

    writer.close()
    write_metrics_csv(result_save_path / "metrics.csv", metrics)
    plot_training_curves(metrics, result_save_path / "loss_accuracy_curve.png")
    return metrics


def plot_training_curves(metrics: list[Metrics], output_path: Path) -> None:
    if not metrics:
        return
    epochs = [row.epoch for row in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row.train_loss for row in metrics], marker="o", label="Train")
    axes[0].plot(epochs, [row.val_loss for row in metrics], marker="s", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row.train_acc for row in metrics], marker="o", label="Train")
    axes[1].plot(epochs, [row.val_acc for row in metrics], marker="s", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    result_save_path: Path,
    show_progress: bool = False,
    verbose: bool = True,
) -> tuple[float, float]:
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="test", disable=not show_progress):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    loss_value = test_loss / max(len(test_loader), 1)
    acc_value = 100.0 * correct_test / max(total_test, 1)
    line = f"Test Loss: {loss_value:.4f}, Test Acc: {acc_value:.2f}%"
    if verbose:
        print(line)
    with (result_save_path / "result.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return loss_value, acc_value


def visualize_filters(model: AlexNet, output_dir: Path, num_maps: int = 8) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_idxs = [0, 3, 6, 8, 10]
    for conv_id, layer_id in enumerate(layer_idxs, start=1):
        filters = model.features[layer_id].weight.detach().cpu()
        num_features = min(num_maps, filters.size(0))
        cols = 4
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < num_features:
                kernel = filters[i, 0]
                ax.imshow(kernel, cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"conv_{conv_id}.png", dpi=180)
        plt.close(fig)


def get_feature_maps(
    model: AlexNet,
    layer_name: str,
    input_image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    feature_maps: dict[str, torch.Tensor] = {}
    handle = None

    def hook(_module: nn.Module, _input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        feature_maps[layer_name] = output.detach().cpu()

    for name, layer in model.named_modules():
        if name == layer_name:
            handle = layer.register_forward_hook(hook)
            break
    if handle is None:
        raise ValueError(f"Layer {layer_name!r} not found")

    model.eval()
    with torch.no_grad():
        model(input_image.unsqueeze(0).to(device))
    handle.remove()
    return feature_maps[layer_name]


def visualize_feature_maps(
    feature_maps: torch.Tensor,
    layer_name: str,
    output_dir: Path,
    num_maps: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_maps = feature_maps.squeeze(0)
    num_features = min(num_maps, feature_maps.size(0))
    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < num_features:
            ax.imshow(feature_maps[i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    safe_layer_name = layer_name.replace(".", "_")
    plt.savefig(output_dir / f"{safe_layer_name}.png", dpi=180)
    plt.close(fig)


def run_visualizations(
    model: AlexNet,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    result_dir: Path,
    verbose: bool = True,
) -> None:
    visualize_filters(model, result_dir / "conv_filter")
    sample_image, _ = dataset[0]
    feature_dir = result_dir / "feature_maps"
    for i, layer in enumerate(model.features):
        if not isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
            continue
        layer_name = f"features.{i}"
        feature_maps = get_feature_maps(model, layer_name, sample_image, device)
        if verbose:
            print(f"Feature map shape from {layer_name}: {tuple(feature_maps.shape)}")
        visualize_feature_maps(feature_maps, layer_name, feature_dir)


def run_hyper_parameter_experiments(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [32, 64, 128]
    dropouts = [0.3, 0.5, 0.7]
    rows: list[dict[str, float]] = []
    result_all_dir = args.result_dir / "result_all"
    result_all_dir.mkdir(parents=True, exist_ok=True)

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout in dropouts:
                if not args.quiet:
                    print(f"Experiment: lr={lr}, batch_size={batch_size}, dropout={dropout}")
                name = f"lr={lr}_batch_size={batch_size}_dropout={dropout}"
                train_loader = make_loader(train_dataset, batch_size, True, args.num_workers)
                val_loader = make_loader(val_dataset, batch_size, False, args.num_workers)
                test_loader = make_loader(test_dataset, batch_size, False, args.num_workers)

                model = build_model(
                    num_classes=10,
                    dropout=dropout,
                    device=device,
                    verbose=not args.quiet,
                )
                optimizer = optim.Adam(model.parameters(), lr=lr)
                metrics = train_and_validate(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    epochs=args.hparam_epochs,
                    log_path=args.log_dir / name,
                    result_save_path=args.result_dir / f"result_{name}",
                    show_progress=args.progress,
                    verbose=not args.quiet,
                )
                test_loss, test_acc = test_model(
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    result_save_path=args.result_dir / f"result_{name}",
                    show_progress=args.progress,
                    verbose=not args.quiet,
                )
                last = metrics[-1]
                rows.append(
                    {
                        "lr": lr,
                        "batch_size": batch_size,
                        "dropout": dropout,
                        "train_loss": last.train_loss,
                        "train_acc": last.train_acc,
                        "val_loss": last.val_loss,
                        "val_acc": last.val_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                    }
                )

    csv_path = result_all_dir / "hparam_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (result_all_dir / "result.txt").open("w", encoding="utf-8") as f:
        for row in rows:
            line = (
                f"lr={row['lr']}, batch_size={row['batch_size']}, dropout={row['dropout']} -> "
                f"Train Loss: {row['train_loss']:.4f}, Train Acc: {row['train_acc']:.2f}, "
                f"Val Loss: {row['val_loss']:.4f}, Val Acc: {row['val_acc']:.2f}, "
                f"Test Loss: {row['test_loss']:.4f}, Test Acc: {row['test_acc']:.2f}"
            )
            if not args.quiet:
                print(line)
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlexNet MNIST experiment")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hparam-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-hparam", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output for Jupyter/Kaggle runs.")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm batch progress bars. Disabled by default to avoid Jupyter/Kaggle IOStream timeouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Using device: {device}")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    if gpu_count and not args.quiet:
        print(f"Detected CUDA GPUs: {gpu_count} - {', '.join(gpu_names)}")

    dataset = load_dataset(args.data_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, args.seed)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    model = build_model(
        num_classes=10,
        dropout=args.dropout,
        device=device,
        verbose=not args.quiet,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    base_result_dir = args.result_dir / "result_base"

    metrics = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        log_path=args.log_dir / "base",
        result_save_path=base_result_dir,
        show_progress=args.progress,
        verbose=not args.quiet,
    )
    test_loss, test_acc = test_model(
        model,
        test_loader,
        criterion,
        device,
        base_result_dir,
        show_progress=args.progress,
        verbose=not args.quiet,
    )

    torch.save(unwrap_model(model).state_dict(), base_result_dir / "alexnet_mnist.pth")
    summary = {
        "device": str(device),
        "cuda_gpu_count": gpu_count,
        "cuda_gpu_names": gpu_names,
        "data_parallel": isinstance(model, nn.DataParallel),
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "base_last_epoch": asdict(metrics[-1]),
        "base_test_loss": test_loss,
        "base_test_acc": test_acc,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (base_result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.skip_visualization:
        run_visualizations(
            unwrap_model(model),
            dataset,
            device,
            args.result_dir,
            verbose=not args.quiet,
        )

    if not args.skip_hparam:
        run_hyper_parameter_experiments(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            criterion=criterion,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
