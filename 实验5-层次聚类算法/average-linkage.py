import os
import numpy as np
from hierarchical import (
    AgglomerativeClustering, load_wine, best_mapping_accuracy, 
    calculate_ari, pca_transform, plot_results, plot_confusion
)

def main():
    df = load_wine()
    X = df.drop(columns=["class"]).values.astype(float)
    y = df["class"].values

    # 1. 预处理
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    print("Pre-processing: Using standardized 13D features for clustering...")

    # 2. 聚类
    print("Running Average Linkage Clustering...")
    clf = AgglomerativeClustering(n_clusters=3, linkage="average")
    labels = clf.fit_predict(X_std)

    # 3. 评估 (ACC 和 ARI)
    acc, mapped_labels = best_mapping_accuracy(y, labels)
    ari = calculate_ari(y, labels)
    
    print("-" * 30)
    print(f"Average Linkage Results:")
    print(f"Accuracy (ACC):        {acc:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print("-" * 30)

    # 4. 可视化
    X_vis = pca_transform(X_std, n_components=2)
    plot_results(X_vis, y, mapped_labels, "Average Linkage", os.path.dirname(__file__))

    # 保存混淆矩阵
    plot_confusion(y, mapped_labels, os.path.join(os.path.dirname(__file__), "average_linkage_confusion.png"), "Average Linkage Confusion Matrix")

if __name__ == "__main__":
    main()