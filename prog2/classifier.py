import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "MNIST"
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "reports" / "figures"


def parse_mnist(minst_file_addr: str | Path = None, flatten: bool = False, one_hot: bool = False) -> np.ndarray:
    """解析 MNIST gzip 二进制文件，返回标签或图片数组。"""
    if minst_file_addr is None:
        raise ValueError("请传入 MNIST 文件地址")

    minst_file_addr = Path(minst_file_addr)
    minst_file_name = os.path.basename(minst_file_addr)

    with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
        minst_file_content = minst_file.read()

    if "label" in minst_file_name:
        data = np.frombuffer(buffer=minst_file_content, dtype=np.uint8, offset=8)
        if one_hot:
            data_zeros = np.zeros(shape=(data.size, 10), dtype=np.uint8)
            data_zeros[np.arange(data.size), data] = 1
            data = data_zeros
    else:
        data = np.frombuffer(buffer=minst_file_content, dtype=np.uint8, offset=16)
        data = data.reshape(-1, 784) if flatten else data.reshape(-1, 28, 28)

    return data


def load_mnist(dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取训练集和测试集。"""
    data_image = parse_mnist(dataset_dir / "train-images-idx3-ubyte.gz", flatten=True)
    data_label = parse_mnist(dataset_dir / "train-labels-idx1-ubyte.gz")
    test_image = parse_mnist(dataset_dir / "t10k-images-idx3-ubyte.gz", flatten=True)
    test_label = parse_mnist(dataset_dir / "t10k-labels-idx1-ubyte.gz")
    return data_image, data_label, test_image, test_label


def binarize_images(images: np.ndarray, threshold: float = 127.5) -> np.ndarray:
    """对图像进行二值化，像素值大于阈值记为 1，否则记为 0。"""
    return (images > threshold).astype(np.uint8)


def train_naive_bayes(
    binary_data_image: np.ndarray,
    data_label: np.ndarray,
    class_num: int = 10,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练 Bernoulli 朴素贝叶斯分类器。"""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    label_count = np.array([np.count_nonzero(data_label == j) for j in range(class_num)], dtype=np.float64)
    p_classability_list = label_count / len(data_label)

    p_condition = np.zeros((class_num, binary_data_image.shape[1]), dtype=np.float64)
    for j in tqdm(range(class_num), desc="Training condition probability"):
        binary_class_images = binary_data_image[data_label == j]
        freq_xi_yj = np.sum(binary_class_images, axis=0)
        p_condition[j, :] = (freq_xi_yj + alpha) / (label_count[j] + 2 * alpha)

    return p_classability_list, p_condition, label_count


def predict(X: np.ndarray, P_classability_list: np.ndarray, P_condition: np.ndarray) -> int:
    """预测单个样本类别,用课件中给出的函数接口。"""
    X = np.asarray(X, dtype=np.uint8)
    eps = np.finfo(np.float64).tiny
    condition = np.clip(P_condition, eps, 1 - eps)
    log_likelihood = np.sum(
        np.where(X == 1, np.log(condition), np.log1p(-condition)),
        axis=1,
    )
    log_posterior = np.log(np.clip(P_classability_list, eps, 1.0)) + log_likelihood
    return int(np.argmax(log_posterior))


def predict_batch(binary_test_image: np.ndarray, priors: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """批量预测测试集类别。"""
    eps = np.finfo(np.float64).tiny
    condition = np.clip(condition, eps, 1 - eps)
    log_condition = np.log(condition)
    log_not_condition = np.log1p(-condition)
    log_priors = np.log(np.clip(priors, eps, 1.0))

    scores = binary_test_image @ log_condition.T + (1 - binary_test_image) @ log_not_condition.T
    scores += log_priors
    return np.argmax(scores, axis=1).astype(np.uint8)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_num: int = 10) -> np.ndarray:
    matrix = np.zeros((class_num, class_num), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def evaluate_accuracy(pred_labels: np.ndarray, test_label: np.ndarray) -> float:
    return float(np.mean(pred_labels == test_label))


def plot_training_sample(binary_data_image: np.ndarray, data_label: np.ndarray, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_file = figure_dir / "train_sample.png"
    plt.figure(figsize=(5.6, 4.6))
    plt.imshow(binary_data_image[0].reshape(28, 28), cmap="gray")
    plt.colorbar()
    plt.title(f"Mnist Visualization: Label {data_label[0]}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return output_file


def plot_prediction_samples(
    binary_test_image: np.ndarray,
    test_label: np.ndarray,
    pred_labels: np.ndarray,
    figure_dir: Path,
    sample_count: int = 12,
) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_file = figure_dir / "prediction_samples.png"
    sample_count = min(sample_count, len(test_label))
    cols = 6
    rows = int(np.ceil(sample_count / cols))

    plt.figure(figsize=(12, 2.25 * rows))
    for i in range(sample_count):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(binary_test_image[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"Predict {pred_labels[i]} / True {test_label[i]}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return output_file


def plot_confusion_matrix(matrix: np.ndarray, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_file = figure_dir / "confusion_matrix.png"
    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Naive Bayes Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    threshold = matrix.max() * 0.58
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > threshold else "black"
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return output_file


def plot_threshold_accuracy(threshold_results: list[dict[str, float]], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_file = figure_dir / "threshold_accuracy.png"
    thresholds = [item["threshold"] for item in threshold_results]
    accuracies = [item["accuracy"] for item in threshold_results]

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(thresholds, accuracies, marker="o", linewidth=2)
    plt.xlabel("Binarization threshold")
    plt.ylabel("Accuracy")
    plt.title("Threshold Influence on MNIST Naive Bayes")
    plt.grid(alpha=0.28)
    for x_value, y_value in zip(thresholds, accuracies):
        plt.text(x_value, y_value + 0.002, f"{y_value:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return output_file


def plot_training_process(summary: dict, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_file = figure_dir / "training_process.png"
    text_lines = [
        "Command: python prog2/classifier.py --threshold 127.5 --threshold-sweep",
        f"Train samples: {summary['train_samples']}, Test samples: {summary['test_samples']}",
        f"Feature dimension: {summary['feature_dim']}, Classes: 10",
        f"Binarization threshold: {summary['threshold']}, Laplace alpha: {summary['alpha']}",
        "Prior probabilities:",
        np.array2string(np.array(summary["priors"]), precision=4, separator=", "),
        f"Accuracy: {summary['accuracy']:.4f}",
    ]

    plt.figure(figsize=(11.5, 4.7))
    plt.axis("off")
    plt.text(
        0.02,
        0.96,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=12,
        linespacing=1.35,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return output_file


def save_predictions(result_dir: Path, pred_labels: np.ndarray, test_label: np.ndarray) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / "predictions.csv"
    with output_file.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_index", "pred_label", "true_label", "correct"])
        for i, (pred_label, true_label) in enumerate(zip(pred_labels, test_label)):
            writer.writerow([i, int(pred_label), int(true_label), int(pred_label == true_label)])
    return output_file


def run_experiment(
    threshold: float = 127.5,
    alpha: float = 1.0,
    threshold_sweep: bool = False,
    result_dir: Path = RESULT_DIR,
    figure_dir: Path = FIGURE_DIR,
) -> dict:
    data_image, data_label, test_image, test_label = load_mnist()
    binary_data_image = binarize_images(data_image, threshold=threshold)
    binary_test_image = binarize_images(test_image, threshold=threshold)

    plot_training_sample(binary_data_image, data_label, figure_dir)

    priors, condition, label_count = train_naive_bayes(binary_data_image, data_label, alpha=alpha)
    pred_labels = predict_batch(binary_test_image, priors, condition)
    accuracy = evaluate_accuracy(pred_labels, test_label)
    matrix = confusion_matrix(test_label, pred_labels)

    plot_prediction_samples(binary_test_image, test_label, pred_labels, figure_dir)
    plot_confusion_matrix(matrix, figure_dir)
    save_predictions(result_dir, pred_labels, test_label)

    threshold_results = []
    if threshold_sweep:
        for item_threshold in [32, 64, 96, 127.5, 160, 192, 224]:
            item_binary_train = binarize_images(data_image, threshold=item_threshold)
            item_binary_test = binarize_images(test_image, threshold=item_threshold)
            item_priors, item_condition, _ = train_naive_bayes(
                item_binary_train,
                data_label,
                alpha=alpha,
            )
            item_pred = predict_batch(item_binary_test, item_priors, item_condition)
            threshold_results.append(
                {
                    "threshold": float(item_threshold),
                    "accuracy": evaluate_accuracy(item_pred, test_label),
                }
            )
        plot_threshold_accuracy(threshold_results, figure_dir)

    summary = {
        "train_samples": int(len(data_label)),
        "test_samples": int(len(test_label)),
        "feature_dim": int(data_image.shape[1]),
        "threshold": float(threshold),
        "alpha": float(alpha),
        "label_count": label_count.astype(int).tolist(),
        "priors": priors.tolist(),
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
        "threshold_results": threshold_results,
    }
    plot_training_process(summary, figure_dir)

    result_dir.mkdir(parents=True, exist_ok=True)
    summary_file = result_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train samples: {summary['train_samples']}, Test samples: {summary['test_samples']}")
    print(f"Threshold: {threshold}, Laplace alpha: {alpha}")
    print("Prior probabilities:", np.array2string(priors, precision=4, separator=", "))
    print(f"Accuracy:{accuracy:.4f}")
    print(f"Results saved to {result_dir}")
    print(f"Figures saved to {figure_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bernoulli Naive Bayes classifier for MNIST.")
    parser.add_argument("--threshold", type=float, default=127.5, help="Binarization threshold")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing coefficient")
    parser.add_argument("--threshold-sweep", action="store_true", help="Evaluate several thresholds")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR, help="Directory for result files")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR, help="Directory for report figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        threshold=args.threshold,
        alpha=args.alpha,
        threshold_sweep=args.threshold_sweep,
        result_dir=args.result_dir,
        figure_dir=args.figure_dir,
    )
