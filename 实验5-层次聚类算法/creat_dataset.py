import os
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from hierarchical_1 import AgglomerativeClustering, best_mapping_accuracy, calculate_ari, plot_results

def save_as_data_file(name, X, y, out_dir='datasets_data'):
    """
    将数据集保存为 .data 文件 (CSV 格式，无表头)。
    格式: x0, x1, label
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 构造文件名
    safe_name = name.replace(' ', '_').replace('/', '_')
    file_path = os.path.join(out_dir, f"{safe_name}.data")
    
    # 合并特征和标签
    # X 是 (n_samples, 2), y 是 (n_samples, )
    # 最终 data 是 (n_samples, 3)
    data = np.hstack([X, y.reshape(-1, 1)])
    
    # 保存为文本文件
    # fmt: 前两列保留6位小数，最后一列为整数
    np.savetxt(file_path, data, delimiter=',', fmt='%.6f,%.6f,%d')
    
    print(f"Dataset saved to: {file_path}")
    return file_path

def run_clustering_and_plot(X, y, dataset_name, out_dir):
    """
    对数据集运行三种 linkage 聚类，并生成可视化图。
    """
    linkages = ['single', 'complete', 'average']
    for linkage in linkages:
        print(f"Running {linkage} linkage on {dataset_name}...")
        clf = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        labels = clf.fit_predict(X)
        
        acc, mapped_labels = best_mapping_accuracy(y, labels)
        ari = calculate_ari(y, labels)
        
        print(f"  {linkage}: ACC={acc:.4f}, ARI={ari:.4f}")
        
        plot_results(X, y, mapped_labels, f"{dataset_name}_{linkage}", out_dir)

def generate_and_export():
    n_samples = 500
    seed = 42
    rng = np.random.RandomState(seed)
    out_dir = 'datasets_data'

    print("Generating datasets...")

    # ==========================================
    # 1. Dataset for Single-linkage: 同心圆
    # ==========================================
    X_single, y_single = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    X_single = StandardScaler().fit_transform(X_single)
    
    save_as_data_file("Target_Single_Circles", X_single, y_single)
    
    # 运行聚类和可视化
    run_clustering_and_plot(X_single, y_single, "Target_Single_Circles", out_dir)

    # ==========================================
    # 2. Dataset for Average-linkage: 紧密簇 + 稀疏簇 + 高密度链条
    # ==========================================
    # 簇1 (Tight): 左下角，紧密
    X_blob1, y_blob1 = datasets.make_blobs(n_samples=100, centers=[[-2, -2]], cluster_std=[0.3], random_state=seed)
    
    # 簇2 (Loose): 右上角，巨大且稀疏 (打击 Complete)
    X_blob2, y_blob2 = datasets.make_blobs(n_samples=300, centers=[[3, 3]], cluster_std=[1.8], random_state=seed)
    y_blob2[:] = 1 
    
    # 桥梁 (Chain): 高密度直线 (打击 Single)
    # 使用 60 个点形成极高密度的链条
    bridge_x = np.linspace(-1.5, 2.0, 60)
    bridge_y = np.linspace(-1.5, 2.0, 60)
    bridge_x += rng.normal(0, 0.02, 60) # 微小抖动
    bridge_y += rng.normal(0, 0.02, 60)
    
    X_bridge = np.c_[bridge_x, bridge_y]
    y_bridge = np.full(60, -1) # 桥视为噪声

    X_avg = np.vstack([X_blob1, X_blob2, X_bridge])
    y_avg = np.concatenate([y_blob1, y_blob2, y_bridge])
    X_avg = StandardScaler().fit_transform(X_avg)
    
    save_as_data_file("Target_Average_DenseChain", X_avg, y_avg)
    
    # 运行聚类和可视化
    run_clustering_and_plot(X_avg, y_avg, "Target_Average_DenseChain", out_dir)

    # ==========================================
    # 3. Dataset for Complete-linkage: 紧密球体 + 强力连接
    # ==========================================
    X_comp, y_comp = datasets.make_blobs(n_samples=n_samples, centers=[[-1.5, 0], [1.5, 0]], cluster_std=0.4, random_state=seed)
    
    # 高密度桥梁
    bridge_x = np.linspace(-1.0, 1.0, 50)
    bridge_y = np.random.normal(0, 0.02, 50) 
    X_bridge = np.c_[bridge_x, bridge_y]
    y_bridge = np.full(50, -1) 

    X_comp = np.vstack([X_comp, X_bridge])
    y_comp = np.concatenate([y_comp, y_bridge])
    X_comp = StandardScaler().fit_transform(X_comp)
    
    save_as_data_file("Target_Complete_DenseBridge", X_comp, y_comp)
    
    # 运行聚类和可视化
    run_clustering_and_plot(X_comp, y_comp, "Target_Complete_DenseBridge", out_dir)

    print("\nAll datasets have been exported to the 'datasets_data' folder.")

if __name__ == "__main__":
    generate_and_export()