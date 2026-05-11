"""Generate LaTeX table fragments for the experiment report."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RESULT_ALL_DIR = PROJECT_DIR / "results" / "result_all"
REPORT_DIR = PROJECT_DIR / "reports"


def fmt_float(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_percent(value: str) -> str:
    return f"{float(value):.2f}\\%"


def generate_hparam_table() -> None:
    csv_paths = sorted(
        RESULT_ALL_DIR.glob("hparam_results_*.csv"),
        key=lambda path: int(path.stem.split("_")[-2]),
    )
    if not csv_paths:
        raise FileNotFoundError(f"No hparam_results_*.csv files found in {RESULT_ALL_DIR}")

    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["index"]))

    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{longtable}{c c c c c c c c c c}",
        r"\caption{不同超参数下的 AlexNet 图像分类结果}\label{tab:hparam}\\",
        r"\toprule",
        r"编号 & 学习率 & Batch size & Dropout & 训练损失 & 训练准确率 & 验证损失 & 验证准确率 & 测试损失 & 测试准确率\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"编号 & 学习率 & Batch size & Dropout & 训练损失 & 训练准确率 & 验证损失 & 验证准确率 & 测试损失 & 测试准确率\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["index"],
                    row["lr"],
                    row["batch_size"],
                    row["dropout"],
                    fmt_float(row["train_loss"]),
                    fmt_percent(row["train_acc"]),
                    fmt_float(row["val_loss"]),
                    fmt_percent(row["val_acc"]),
                    fmt_float(row["test_loss"]),
                    fmt_percent(row["test_acc"]),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize", r"\setlength{\tabcolsep}{6pt}"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "hparam_table.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    generate_hparam_table()
