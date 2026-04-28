import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "fooddata.xlsx"
RESULT_DIR = BASE_DIR / "results"


def kMeansInitCentroids(X, k, random_state=None):
    """随机选择 k 个样本作为初始聚类中心。"""
    X = np.asarray(X, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(X):
        raise ValueError("k cannot be greater than the number of samples")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def findClosestCentroids(X, centroids):
    """返回每个样本距离最近的聚类中心编号。"""
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    squared_distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    return np.argmin(squared_distances, axis=1).astype(int)


def computeCentroids(X, idx, k=None, old_centroids=None, random_state=None):
    """根据当前样本分配结果重新计算聚类中心。"""
    X = np.asarray(X, dtype=float)
    idx = np.asarray(idx, dtype=int)
    if k is None:
        k = int(idx.max()) + 1

    centroids = np.zeros((k, X.shape[1]), dtype=float)
    rng = np.random.default_rng(random_state)

    for cluster_id in range(k):
        members = X[idx == cluster_id]
        if len(members) > 0:
            centroids[cluster_id] = members.mean(axis=0)
        elif old_centroids is not None:
            centroids[cluster_id] = old_centroids[cluster_id]
        else:
            centroids[cluster_id] = X[rng.integers(0, len(X))]

    return centroids


def calculate_inertia(X, idx, centroids):
    """计算聚类目标函数值，即样本到所属中心的平方距离和。"""
    X = np.asarray(X, dtype=float)
    return float(np.sum((X - centroids[idx]) ** 2))


def k_means(X, k, max_iters=500, n_init=10, tol=1e-6, random_state=42):
    """手写 K-means，实现多次随机初始化并返回最优结果。"""
    X = np.asarray(X, dtype=float)
    best_idx = None
    best_centroids = None
    best_inertia = np.inf
    rng = np.random.default_rng(random_state)

    for _ in range(n_init):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        centroids = kMeansInitCentroids(X, k, random_state=seed)

        for _ in range(max_iters):
            idx = findClosestCentroids(X, centroids)
            new_centroids = computeCentroids(
                X,
                idx,
                k=k,
                old_centroids=centroids,
                random_state=seed,
            )

            center_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if center_shift <= tol:
                break

        idx = findClosestCentroids(X, centroids)
        inertia = calculate_inertia(X, idx, centroids)
        if inertia < best_inertia:
            best_idx = idx
            best_centroids = centroids.copy()
            best_inertia = inertia

    return best_idx, best_centroids


def load_food_data(data_file=DATA_FILE):
    """读取食物营养数据，并返回原始表与可聚类的数值特征。"""
    df = pd.read_excel(data_file)
    df = df.dropna().reset_index(drop=True)

    feature_df = df.drop(columns=["食物名", "序号"], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    valid_rows = feature_df.notna().all(axis=1)
    df = df.loc[valid_rows].reset_index(drop=True)
    feature_df = feature_df.loc[valid_rows].reset_index(drop=True)
    return df, feature_df


def preprocess_features(feature_df):
    """先标准化再归一化，减小不同营养指标量纲差异的影响。"""
    z_scaler = preprocessing.StandardScaler()
    data_z = z_scaler.fit_transform(feature_df)

    minmax_scaler = preprocessing.MinMaxScaler()
    return minmax_scaler.fit_transform(data_z)


def build_result_frame(df, labels):
    result = df.copy()
    result["类_别"] = labels.astype(int)
    return result.sort_values(["类_别", "食物名"]).reset_index(drop=True)


def save_cluster_result(file_path, result_df):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(file_path, index=False, encoding="utf-8-sig")


def savaData(filePath, data):
    """保留原参考代码中的函数名，兼容旧调用。"""
    Path(filePath).write_text(str(data), encoding="utf-8")


def run_all_k(data_file=DATA_FILE, output_dir=RESULT_DIR, min_k=2, max_k=12, random_state=42):
    df, feature_df = load_food_data(data_file)
    data = preprocess_features(feature_df)

    summaries = []
    for k in range(min_k, max_k + 1):
        labels, centroids = k_means(data, k, max_iters=500, n_init=20, random_state=random_state)
        result_df = build_result_frame(df, labels)
        output_file = Path(output_dir) / f"kmeans_result{k}.csv"
        save_cluster_result(output_file, result_df)

        inertia = calculate_inertia(data, labels, centroids)
        summaries.append({"k": k, "inertia": inertia, "output": str(output_file)})
        print(f"k={k}, inertia={inertia:.6f}, saved to {output_file}")

    return pd.DataFrame(summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="Manual K-means clustering for food nutrition data.")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Path to fooddata.xlsx")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR, help="Directory for result CSV files")
    parser.add_argument("--min-k", type=int, default=2, help="Minimum k")
    parser.add_argument("--max-k", type=int, default=12, help="Maximum k")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_k(
        data_file=args.data,
        output_dir=args.output_dir,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )
