import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from kmeans import (
    DATA_FILE,
    RESULT_DIR,
    build_result_frame,
    calculate_inertia,
    load_food_data,
    preprocess_features,
    save_cluster_result,
)


def run_sklearn_kmeans(data_file=DATA_FILE, output_dir=RESULT_DIR, min_k=2, max_k=12, random_state=42):
    """使用 sklearn 的 KMeans 对同一份食物营养数据进行聚类。"""
    df, feature_df = load_food_data(data_file)
    data = preprocess_features(feature_df)

    summaries = []
    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=500,
            random_state=random_state,
        )
        labels = model.fit_predict(data)

        result_df = build_result_frame(df, labels)
        output_file = Path(output_dir) / f"kmeans_sklearn_result{k}.csv"
        save_cluster_result(output_file, result_df)

        inertia = calculate_inertia(data, labels, model.cluster_centers_)
        summaries.append({"k": k, "inertia": inertia, "output": str(output_file)})
        print(f"k={k}, inertia={inertia:.6f}, saved to {output_file}")

    return pd.DataFrame(summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="sklearn KMeans clustering for food nutrition data.")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Path to fooddata.xlsx")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR, help="Directory for result CSV files")
    parser.add_argument("--min-k", type=int, default=2, help="Minimum k")
    parser.add_argument("--max-k", type=int, default=12, help="Maximum k")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sklearn_kmeans(
        data_file=args.data,
        output_dir=args.output_dir,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )
