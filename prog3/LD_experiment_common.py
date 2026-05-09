from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "AI_LD_experiment_dataset"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    method: str
    setting: str
    parameters: str
    train_accuracy: float
    test_accuracy: float


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_adaboost(
    *,
    max_depth: int = 1,
    criterion: str = "gini",
    n_estimators: int = 50,
    learning_rate: float = 0.5,
    random_state: int | None = 1,
) -> AdaBoostClassifier:
    base_model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
    )
    return AdaBoostClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def make_fisher(
    *,
    solver: str = "svd",
    shrinkage: str | float | None = None,
    n_components: int | None = 1,
    standardize: bool = False,
) -> BaseEstimator:
    model = LinearDiscriminantAnalysis(
        solver=solver,
        shrinkage=shrinkage,
        n_components=n_components if solver == "svd" else None,
    )
    if standardize:
        return make_pipeline(StandardScaler(), model)
    return model


def make_svm(
    *,
    C: float = 10,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    decision_function_shape: str = "ovr",
    class_weight=None,
    degree: int = 3,
) -> BaseEstimator:
    return make_pipeline(
        StandardScaler(),
        svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            decision_function_shape=decision_function_shape,
            class_weight=class_weight,
            degree=degree,
        ),
    )


def make_stump(*, max_depth: int = 1, criterion: str = "gini") -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=1)


def evaluate(
    *,
    dataset: str,
    method: str,
    setting: str,
    parameters: str,
    model: BaseEstimator,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[ExperimentResult, np.ndarray]:
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    result = ExperimentResult(
        dataset=dataset,
        method=method,
        setting=setting,
        parameters=parameters,
        train_accuracy=accuracy_score(y_train, train_pred),
        test_accuracy=accuracy_score(y_test, test_pred),
    )
    return result, test_pred


def save_results_csv(results: Iterable[ExperimentResult], filename: str) -> pd.DataFrame:
    ensure_output_dirs()
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(RESULTS_DIR / filename, index=False, encoding="utf-8-sig")
    return df


def print_results(results: Iterable[ExperimentResult]) -> None:
    for r in results:
        print(
            f"{r.dataset} | {r.method} | {r.setting} | "
            f"训练集分类准确率:{r.train_accuracy:.4f} | "
            f"测试集分类准确率:{r.test_accuracy:.4f} | 参数:{r.parameters}"
        )


def plot_accuracy_bars(df: pd.DataFrame, title: str, output: Path) -> None:
    ensure_output_dirs()
    base = df[df["setting"] == "默认参数"].copy()
    label_map = {"弱分类器": "Decision tree"}
    labels = [label_map.get(method, method) for method in base["method"].tolist()]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(x - width / 2, base["train_accuracy"], width, label="Train")
    plt.bar(x + width / 2, base["test_accuracy"], width, label="Test")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    output: Path,
) -> None:
    ensure_output_dirs()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close(fig)
