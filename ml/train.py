import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

try:
    from ml.model import QuickDrawCNN
except ImportError:
    from model import QuickDrawCNN

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
NPY_DIR_DEFAULT = ARTIFACTS_DIR / "quickdraw_npy"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
TRAIN_CONFIG_PATH = ARTIFACTS_DIR / "train_config.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SPLIT_INDEX_PATH = ARTIFACTS_DIR / "split_indices.npz"
CLASSES_PATH = Path(__file__).resolve().parent / "classes.json"


@dataclass
class TrainConfig:
    train_per_class: int = 4000
    val_per_class: int = 500
    test_per_class: int = 500
    batch_size: int = 128
    epochs: int = 20
    patience: int = 4
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


class StrokeWidthJitter:
    def __init__(self, p: float = 0.35) -> None:
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        x = tensor.unsqueeze(0)
        if random.random() < 0.5:
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        else:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x.squeeze(0).clamp(0.0, 1.0)


class QuickDrawBitmapDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self.images = images
        self.labels = labels

        if augment:
            self.transform = T.Compose(
                [
                    T.ToPILImage(),
                    T.RandomAffine(
                        degrees=15,
                        translate=(0.12, 0.12),
                        scale=(0.9, 1.1),
                        fill=0,
                    ),
                    T.ToTensor(),
                    StrokeWidthJitter(),
                ]
            )
        else:
            self.transform = T.Compose([T.ToTensor()])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        tensor = self.transform(image)
        return tensor, torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QuickDraw CNN (20 classes)")
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--npy-dir", type=str, default=str(NPY_DIR_DEFAULT))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes() -> List[str]:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_device(option: str) -> torch.device:
    if option == "cpu":
        return torch.device("cpu")
    if option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split_data(
    class_names: List[str],
    npy_dir: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    total = train_count + val_count + test_count
    rng = np.random.default_rng(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    split_indices: Dict[str, np.ndarray] = {}

    for class_idx, class_name in enumerate(class_names):
        file_path = npy_dir / f"{class_name}.npy"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing class file: {file_path}. Run python ml/download_quickdraw.py first."
            )

        data = np.load(file_path)
        if len(data) < total:
            raise ValueError(
                f"Class '{class_name}' has {len(data)} samples, requires at least {total}."
            )

        indices = rng.permutation(len(data))[:total]
        split_indices[f"{class_name}_train"] = indices[:train_count]
        split_indices[f"{class_name}_val"] = indices[train_count : train_count + val_count]
        split_indices[f"{class_name}_test"] = indices[train_count + val_count :]

        selected = data[indices].reshape(-1, 28, 28).astype(np.uint8)

        x_train.append(selected[:train_count])
        y_train.append(np.full((train_count,), class_idx, dtype=np.int64))

        x_val.append(selected[train_count : train_count + val_count])
        y_val.append(np.full((val_count,), class_idx, dtype=np.int64))

        x_test.append(selected[train_count + val_count :])
        y_test.append(np.full((test_count,), class_idx, dtype=np.int64))

    result = {
        "x_train": np.concatenate(x_train, axis=0),
        "y_train": np.concatenate(y_train, axis=0),
        "x_val": np.concatenate(x_val, axis=0),
        "y_val": np.concatenate(y_val, axis=0),
        "x_test": np.concatenate(x_test, axis=0),
        "y_test": np.concatenate(y_test, axis=0),
    }

    train_perm = rng.permutation(len(result["y_train"]))
    val_perm = rng.permutation(len(result["y_val"]))
    test_perm = rng.permutation(len(result["y_test"]))

    result["x_train"] = result["x_train"][train_perm]
    result["y_train"] = result["y_train"][train_perm]
    result["x_val"] = result["x_val"][val_perm]
    result["y_val"] = result["y_val"][val_perm]
    result["x_test"] = result["x_test"][test_perm]
    result["y_test"] = result["y_test"][test_perm]

    return result, split_indices


def save_data_visuals(class_names: List[str], split_data: Dict[str, np.ndarray]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    counts = [
        int((split_data["y_train"] == i).sum())
        + int((split_data["y_val"] == i).sum())
        + int((split_data["y_test"] == i).sum())
        for i in range(len(class_names))
    ]

    plt.figure(figsize=(13, 5))
    plt.bar(class_names, counts, color="#f97316")
    plt.title("QuickDraw Class Distribution")
    plt.ylabel("Samples per class")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    for idx, ax in enumerate(axes.flatten()):
        sample_idx = np.where(split_data["y_train"] == idx)[0][0]
        ax.imshow(split_data["x_train"][sample_idx], cmap="gray")
        ax.set_title(class_names[idx])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sample_grid.png", dpi=180)
    plt.close()


def build_loaders(split_data: Dict[str, np.ndarray], cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = QuickDrawBitmapDataset(split_data["x_train"], split_data["y_train"], augment=True)
    val_ds = QuickDrawBitmapDataset(split_data["x_val"], split_data["y_val"], augment=False)
    test_ds = QuickDrawBitmapDataset(split_data["x_test"], split_data["y_test"], augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds, gts = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        pred = torch.argmax(logits, dim=1)
        preds.extend(pred.detach().cpu().numpy().tolist())
        gts.extend(batch_y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(gts, preds)
    return avg_loss, float(acc)


def evaluate_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            gts.append(batch_y.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def save_curves(history: Dict[str, List[float]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4.5))
    plt.plot(epochs, history["train_loss"], label="train_loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="val_loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(epochs, history["train_acc"], label="train_acc", marker="o")
    plt.plot(epochs, history["val_acc"], label="val_acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(preds: np.ndarray, gts: np.ndarray, class_names: List[str]) -> None:
    cm = confusion_matrix(gts, preds, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(13, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Oranges", ax=ax, xticks_rotation=35, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close()


def measure_latency_ms(model: nn.Module, device: torch.device, repeats: int = 200) -> float:
    model.eval()
    sample = torch.rand(1, 1, 28, 28, device=device)

    for _ in range(20):
        _ = model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return ((perf_counter() - t0) * 1000.0) / repeats


def train(cfg: TrainConfig, npy_dir: Path) -> None:
    set_seed(cfg.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_classes()
    print(f"Loaded {len(class_names)} classes.")

    split_data, split_indices = load_split_data(
        class_names,
        npy_dir=npy_dir,
        train_count=cfg.train_per_class,
        val_count=cfg.val_per_class,
        test_count=cfg.test_per_class,
        seed=cfg.seed,
    )
    np.savez(SPLIT_INDEX_PATH, **split_indices)
    save_data_visuals(class_names, split_data)

    device = choose_device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_loaders(split_data, cfg)
    model = QuickDrawCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)

    test_preds, test_gts = evaluate_predictions(model, test_loader, device)
    test_acc = float(accuracy_score(test_gts, test_preds))

    recalls = {}
    for idx, class_name in enumerate(class_names):
        gt_mask = test_gts == idx
        if gt_mask.sum() == 0:
            recalls[class_name] = 0.0
        else:
            recalls[class_name] = float((test_preds[gt_mask] == idx).mean())

    save_curves(history)
    save_confusion_matrix(test_preds, test_gts, class_names)

    latency_ms = measure_latency_ms(model, device=device, repeats=200)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_val_acc": float(best_val_acc),
            "test_acc": test_acc,
        },
        MODEL_PATH,
    )

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open(TRAIN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    metrics = {
        "best_val_acc": float(best_val_acc),
        "test_acc": test_acc,
        "per_class_recall": recalls,
        "mean_inference_latency_ms": float(latency_ms),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nArtifacts saved:")
    print(f"- model: {MODEL_PATH}")
    print(f"- class names: {CLASS_NAMES_PATH}")
    print(f"- train config: {TRAIN_CONFIG_PATH}")
    print(f"- metrics: {METRICS_PATH}")
    print(f"- split indices: {SPLIT_INDEX_PATH}")
    print(f"- figures dir: {FIGURES_DIR}")

    if best_val_acc < 0.80:
        print("[WARN] Validation accuracy is below 0.80 target. Tune epochs/augmentation/samples.")
    if latency_ms > 120:
        print("[WARN] Inference latency is above 120ms target. Consider smaller model or CPU optimization.")


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    train(cfg=cfg, npy_dir=Path(args.npy_dir))


if __name__ == "__main__":
    main()
