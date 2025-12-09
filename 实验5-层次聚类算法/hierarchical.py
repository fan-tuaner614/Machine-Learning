import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import permutations

class AgglomerativeClustering:
    def __init__(self, n_clusters=3, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X):
        clusters = [[i] for i in range(len(X))]
        n_samples = len(X)
        
        # 计算距离矩阵
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d

        # 循环合并
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            cluster_a_idx, cluster_b_idx = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._calculate_distance(clusters[i], clusters[j], dist_matrix)
                    if d < min_dist:
                        min_dist = d
                        cluster_a_idx, cluster_b_idx = i, j

            clusters[cluster_a_idx].extend(clusters[cluster_b_idx])
            clusters.pop(cluster_b_idx)

        labels = np.zeros(n_samples, dtype=int)
        for c_id, indices in enumerate(clusters):
            for idx in indices:
                labels[idx] = c_id
        return labels

    def _calculate_distance(self, c1, c2, dist_matrix):
        if self.linkage == "single":
            min_d = float('inf')
            for i in c1:
                for j in c2:
                    if dist_matrix[i, j] < min_d:
                        min_d = dist_matrix[i, j]
            return min_d
        elif self.linkage == "complete":
            max_d = -1.0
            for i in c1:
                for j in c2:
                    if dist_matrix[i, j] > max_d:
                        max_d = dist_matrix[i, j]
            return max_d
        elif self.linkage == "average":
            total_dist = 0.0
            count = len(c1) * len(c2)
            for i in c1:
                for j in c2:
                    total_dist += dist_matrix[i, j]
            return total_dist / count if count > 0 else 0
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

# --- 通用工具函数 ---

def load_wine(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "wine.data")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find wine.data at {path}")
    cols = ["class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
            "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
            "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline"]
    df = pd.read_csv(path, header=None, names=cols)
    return df

def pca_transform(X, n_components=2):
    cov_mat = np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
    sorted_indices = np.argsort(eigen_vals)[::-1]
    topk_vecs = eigen_vecs[:, sorted_indices[:n_components]]
    return np.dot(X, topk_vecs)

def best_mapping_accuracy(true, pred):
    """ 计算普通准确率 (Accuracy) """
    true = np.asarray(true).astype(int)
    pred = np.asarray(pred)
    clusters = np.unique(pred)
    classes = np.unique(true)
    n_clusters = len(clusters)
    n_classes = len(classes)

    cm = np.zeros((n_clusters, n_classes), dtype=int)
    for i, c in enumerate(clusters):
        for j, cj in enumerate(classes):
            cm[i, j] = np.sum((pred == c) & (true == cj))

    if n_clusters <= n_classes:
        best = -1
        best_perm = None
        for perm in permutations(range(n_classes), n_clusters):
            s = sum(cm[i, perm[i]] for i in range(n_clusters))
            if s > best:
                best = s
                best_perm = perm
        mapped = np.empty_like(pred)
        for i, c in enumerate(clusters):
            mapped[pred == c] = classes[best_perm[i]]
        return best / len(true), mapped
    
    return 0.0, pred

def calculate_ari(true, pred):
    """
    手动实现 Adjusted Rand Index (ARI)
    无需 sklearn，纯 NumPy 实现
    """
    # 确保输入是整数数组
    true = np.asarray(true)
    pred = np.asarray(pred)
    
    # 获取唯一的类别和簇
    classes = np.unique(true)
    clusters = np.unique(pred)
    
    # 1. 构建列联表 (Contingency Table)
    # nij[i, j] 表示同时属于 真实类i 和 预测簇j 的样本数
    tp_plus_fp = np.zeros(len(clusters)) # row sums (a_i)
    tp_plus_fn = np.zeros(len(classes))  # col sums (b_j)
    nij = np.zeros((len(clusters), len(classes)))
    
    for i, c in enumerate(clusters):
        for j, k in enumerate(classes):
            # 计算交集大小
            count = np.sum((pred == c) & (true == k))
            nij[i, j] = count
            
    # 计算行和列的求和
    a_i = np.sum(nij, axis=1) # 预测簇的样本数
    b_j = np.sum(nij, axis=0) # 真实类的样本数
    n = len(true)
    
    # 2. 定义组合数计算函数 C(n, 2) = n*(n-1)/2
    def comb2(x):
        return x * (x - 1) / 2
        
    # 3. 计算各项指标
    sum_nij_comb = np.sum(comb2(nij))          # Index (RI 分子的一部分)
    sum_a_comb = np.sum(comb2(a_i))            # sum(comb2(rows))
    sum_b_comb = np.sum(comb2(b_j))            # sum(comb2(cols))
    total_comb = comb2(n)                      # C(N, 2)
    
    # 4. 套用 ARI 公式
    # Expected Index = (sum_a * sum_b) / total_comb
    expected_index = (sum_a_comb * sum_b_comb) / total_comb
    
    # Max Index = (sum_a + sum_b) / 2
    max_index = (sum_a_comb + sum_b_comb) / 2
    
    # 避免分母为0
    if max_index == expected_index:
        return 0.0
        
    ari = (sum_nij_comb - expected_index) / (max_index - expected_index)
    return ari

def plot_results(X_vis, y_true, y_pred, algorithm_name, out_dir):
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].set_title("True Labels")
    for c in np.unique(y_true):
        mask = y_true == c
        axes[0].scatter(X_vis[mask, 0], X_vis[mask, 1], s=30, color=cmap(int(c)-1), label=f"Class {c}")
    axes[0].legend()

    axes[1].set_title(f"{algorithm_name} Clusters")
    for c in np.unique(y_pred):
        mask = y_pred == c
        axes[1].scatter(X_vis[mask, 0], X_vis[mask, 1], s=30, color=cmap(int(c)), label=f"Cluster {c}")
    axes[1].legend()

    plt.suptitle(f"{algorithm_name} on Wine (PCA Preprocessed)")
    fig_path = os.path.join(out_dir, f"{algorithm_name.lower().replace(' ', '_')}_result.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved scatter plot to {fig_path}")
    plt.close()


def plot_confusion(true, pred_mapped, outpath, title="Confusion"):
    classes = sorted(np.unique(true))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, c in enumerate(classes):
        for j, cj in enumerate(classes):
            cm[i, j] = np.sum((true == c) & (pred_mapped == cj))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(n):
        for j in range(n):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)