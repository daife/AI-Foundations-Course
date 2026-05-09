from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from LD_experiment_common import (
    DATA_ROOT,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    evaluate,
    make_adaboost,
    make_fisher,
    make_stump,
    make_svm,
    plot_accuracy_bars,
    plot_confusion,
    print_results,
    save_results_csv,
)


def load_npha() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dataset_path = DATA_ROOT / "NPHA" / "doctor-visits-dataset.csv"
    dataset = pd.read_csv(dataset_path)
    array = np.array(dataset)
    y = array[:, 0].astype(int)
    x = array[:, 1:].astype(float)
    return x, y, dataset


def run_npha_experiment() -> pd.DataFrame:
    ensure_output_dirs()
    x, y, dataset = load_npha()
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
            "solver='svd', shrinkage=None, n_components=1",
            make_fisher(solver="svd", shrinkage=None, n_components=1),
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
            "StandardScaler + solver='lsqr', shrinkage=0.9",
            make_fisher(solver="lsqr", shrinkage=0.9, standardize=True),
        ),
        (
            "SVM",
            "默认参数",
            "StandardScaler + C=10, kernel='rbf', gamma='scale'",
            make_svm(C=10, kernel="rbf", gamma="scale"),
        ),
        (
            "SVM",
            "调参一",
            "StandardScaler + C=0.1, kernel='rbf', gamma=0.01, class_weight='balanced'",
            make_svm(C=0.1, kernel="rbf", gamma=0.01, class_weight="balanced"),
        ),
        (
            "SVM",
            "调参二",
            "StandardScaler + C=0.5, kernel='sigmoid', gamma=0.003, class_weight='balanced'",
            make_svm(C=0.5, kernel="sigmoid", gamma=0.003, class_weight="balanced"),
        ),
        (
            "弱分类器",
            "默认参数",
            "DecisionTreeClassifier(max_depth=1, criterion='gini')",
            make_stump(max_depth=1, criterion="gini"),
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
            "tree(max_depth=2, criterion='entropy'), n_estimators=200, learning_rate=0.5",
            make_adaboost(max_depth=2, criterion="entropy", n_estimators=200, learning_rate=0.5),
        ),
        (
            "Adaboost",
            "调参二",
            "tree(max_depth=3, criterion='entropy'), n_estimators=200, learning_rate=1.0",
            make_adaboost(max_depth=3, criterion="entropy", n_estimators=200, learning_rate=1.0),
        ),
    ]

    results = []
    predictions = {}
    for method, setting, parameters, model in specs:
        result, test_pred = evaluate(
            dataset="NPHA",
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

    df = save_results_csv(results, "npha_results.csv")
    print(f"数据集: {dataset.shape[0]} 条样本, {dataset.shape[1] - 1} 个特征")
    print(f"划分: 训练集 {len(y_train)} 条, 测试集 {len(y_test)} 条")
    print_results(results)

    best = (
        df[df["method"].isin(["Fisher", "SVM", "Adaboost"])]
        .sort_values(["test_accuracy", "train_accuracy"], ascending=False)
        .iloc[0]
    )
    best_key = (best["method"], best["setting"])
    plot_accuracy_bars(df, "NPHA default-parameter accuracy", FIGURES_DIR / "npha_accuracy.png")
    plot_confusion(
        y_test,
        predictions[best_key],
        ["<4", ">=4"],
        f"NPHA confusion matrix: {best_key[0]} tuned",
        FIGURES_DIR / "npha_confusion_matrix.png",
    )
    df.to_json(RESULTS_DIR / "npha_results.json", orient="records", force_ascii=False, indent=2)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 task 1: NPHA linear discrimination.")
    parser.parse_args()
    run_npha_experiment()


if __name__ == "__main__":
    main()
