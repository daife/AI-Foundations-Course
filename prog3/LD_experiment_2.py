from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from LD_experiment_common import (
    DATA_ROOT,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    evaluate,
    make_adaboost,
    make_fisher,
    make_svm,
    plot_accuracy_bars,
    plot_confusion,
    print_results,
    save_results_csv,
)


def extract_rgb_histogram(image_path: str) -> np.ndarray:
    image = imread(image_path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    img_resized = resize(image, (64, 64, 3), anti_aliasing=True, preserve_range=True)
    img_scaled = np.clip(img_resized, 0, 255).astype("uint8")
    r = np.bincount(img_scaled[:, :, 0].flatten(), minlength=256)
    g = np.bincount(img_scaled[:, :, 1].flatten(), minlength=256)
    b = np.bincount(img_scaled[:, :, 2].flatten(), minlength=256)
    return np.concatenate((r, g, b), dtype="int64")


def load_cat_breeds(rebuild_cache: bool = False) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    ensure_output_dirs()
    cache_path = RESULTS_DIR / "cat_features.npz"
    if cache_path.exists() and not rebuild_cache:
        cached = np.load(cache_path, allow_pickle=True)
        return (
            cached["x"],
            cached["y"],
            cached["class_names"].tolist(),
            cached["image_paths"].tolist(),
        )

    filepath = DATA_ROOT / "cat-breeds"
    file_names = sorted([f for f in os.listdir(filepath) if (filepath / f).is_dir()])
    x, y, image_paths = [], [], []
    for f in file_names:
        img_path = filepath / f
        for img in sorted(os.listdir(img_path)):
            full_path = img_path / img
            feature = extract_rgb_histogram(str(full_path))
            x.append(feature)
            y.append(file_names.index(f))
            image_paths.append(str(full_path))

    x_array = np.array(x)
    y_array = np.array(y)
    np.savez_compressed(
        cache_path,
        x=x_array,
        y=y_array,
        class_names=np.array(file_names),
        image_paths=np.array(image_paths),
    )
    return x_array, y_array, file_names, image_paths


def plot_cat_samples(image_paths: list[str], y: np.ndarray, class_names: list[str]) -> None:
    ensure_output_dirs()
    chosen = []
    for class_idx in range(len(class_names)):
        indices = np.where(y == class_idx)[0][:3]
        chosen.extend(indices.tolist())

    fig, axes = plt.subplots(len(class_names), 3, figsize=(7.0, 5.8))
    for ax, idx in zip(axes.flat, chosen):
        img = imread(image_paths[idx])
        ax.imshow(img)
        ax.set_title(class_names[y[idx]], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cat_samples.png", dpi=180)
    plt.close(fig)


def run_cat_experiment() -> np.ndarray:
    ensure_output_dirs()
    x, y, class_names, image_paths = load_cat_breeds()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=1,
    )

    specs = [
        (
            "Fisher",
            "默认参数",
            "StandardScaler + solver='svd', shrinkage=None, n_components=2",
            make_fisher(solver="svd", shrinkage=None, n_components=2, standardize=True),
        ),
        (
            "Fisher",
            "调参一",
            "StandardScaler + solver='lsqr', shrinkage='auto'",
            make_fisher(solver="lsqr", shrinkage="auto", standardize=True),
        ),
        (
            "Fisher",
            "调参二",
            "StandardScaler + solver='eigen', shrinkage=0.5",
            make_fisher(solver="eigen", shrinkage=0.5, standardize=True),
        ),
        (
            "SVM",
            "默认参数",
            "StandardScaler + C=10, kernel='rbf', gamma='scale', decision_function_shape='ovr'",
            make_svm(C=10, kernel="rbf", gamma="scale", decision_function_shape="ovr"),
        ),
        (
            "SVM",
            "调参一",
            "StandardScaler + C=0.1, kernel='linear', decision_function_shape='ovr'",
            make_svm(C=0.1, kernel="linear", gamma="scale", decision_function_shape="ovr"),
        ),
        (
            "SVM",
            "调参二",
            "StandardScaler + C=1, kernel='poly', degree=2, decision_function_shape='ovo'",
            make_svm(C=1, kernel="poly", gamma="scale", degree=2, decision_function_shape="ovo"),
        ),
        (
            "Adaboost",
            "默认参数",
            "stump(max_depth=1, criterion='gini'), n_estimators=50, learning_rate=0.5",
            make_adaboost(max_depth=1, criterion="gini", n_estimators=50, learning_rate=0.5),
        ),
        (
            "Adaboost",
            "调参一",
            "tree(max_depth=2, criterion='gini'), n_estimators=100, learning_rate=0.5",
            make_adaboost(max_depth=2, criterion="gini", n_estimators=100, learning_rate=0.5),
        ),
        (
            "Adaboost",
            "调参二",
            "tree(max_depth=2, criterion='entropy'), n_estimators=100, learning_rate=0.2",
            make_adaboost(max_depth=2, criterion="entropy", n_estimators=100, learning_rate=0.2),
        ),
    ]

    results = []
    predictions = {}
    for method, setting, parameters, model in specs:
        result, test_pred = evaluate(
            dataset="cat-breeds",
            method=method,
            setting=setting,
            parameters=parameters,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)
        predictions[(method, setting)] = test_pred

    df = save_results_csv(results, "cat_results.csv")
    print(f"数据集: {len(y)} 张图像, 特征维度 {x.shape[1]}, 类别 {class_names}")
    print(f"划分: 训练集 {len(y_train)} 张, 测试集 {len(y_test)} 张")
    print_results(results)

    best = df.sort_values(["test_accuracy", "train_accuracy"], ascending=False).iloc[0]
    best_key = (best["method"], best["setting"])
    plot_accuracy_bars(df, "Cat-breeds default-parameter accuracy", FIGURES_DIR / "cat_accuracy.png")
    plot_confusion(
        y_test,
        predictions[best_key],
        class_names,
        f"Cat-breeds confusion matrix: {best_key[0]} tuned",
        FIGURES_DIR / "cat_confusion_matrix.png",
    )
    plot_cat_samples(image_paths, y, class_names)
    df.to_json(RESULTS_DIR / "cat_results.json", orient="records", force_ascii=False, indent=2)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 task 2: cat-breeds classification.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild RGB histogram features.")
    args = parser.parse_args()
    if args.rebuild_cache:
        cache_path = RESULTS_DIR / "cat_features.npz"
        if cache_path.exists():
            cache_path.unlink()
    run_cat_experiment()


if __name__ == "__main__":
    main()
