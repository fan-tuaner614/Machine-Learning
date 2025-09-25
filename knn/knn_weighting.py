
import numpy as np
import time

# 引入 scikit-learn 用于计算评估指标
from sklearn import metrics

# 引入 matplotlib 用于绘图
import matplotlib.pyplot as plt
import matplotlib
# 解决matplotlib中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False 

def load_and_process_data(file_path):
    """从指定路径加载semeion数据集，并进行处理。"""
    print(f"正在从 '{file_path}' 加载数据...")
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{file_path}'。")
        return None, None
    features = data[:, :256]
    labels_one_hot = data[:, 256:]
    labels = np.argmax(labels_one_hot, axis=1)
    print("数据加载和处理完成。")
    print(f"总样本数: {features.shape[0]}")
    return features, labels

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

class KNN:
    def __init__(self, k=5, verbose=True):
        self.k = k
        self.epsilon = 1e-6 # 防止除以零
        if verbose: print(f"kNN分类器已创建，k值为 {self.k}。")
    
    def fit(self, X_train, y_train, verbose=True):
        self.X_train = X_train
        self.y_train = y_train
        if verbose: print("模型'训练'完成（数据已存储）。")

    def predict_single(self, x_test_point):
        distances = [euclidean_distance(x_test_point, x_train_point) for x_train_point in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        # 加权投票 (距离越近，权重越高)
        k_nearest_distances = [distances[i] for i in k_nearest_indices]
        votes = {}
        for i in range(self.k):
            label = k_nearest_labels[i]
            distance = k_nearest_distances[i]
            weight = 1.0 / (distance + self.epsilon)
            votes[label] = votes.get(label, 0) + weight
        if not votes: return None
        return max(votes, key=votes.get)


def calculate_cen(y_true, y_pred):
    """
    根据真实标签和预测标签计算混淆熵 (Confusion Entropy, CEN)。
    CEN 的值越小，表示分类越明确，混淆程度越低。
    """
    num_classes = len(np.unique(y_true))
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    total_cen = 0.0
    total_samples = len(y_true)
    for i in range(num_classes):
        row = conf_matrix[i, :]
        row_sum = np.sum(row)
        if row_sum == 0: continue
        class_entropy = 0.0
        for count in row:
            if count > 0:
                prob = count / row_sum
                class_entropy -= prob * np.log2(prob)
        total_cen += (row_sum / total_samples) * class_entropy
    return total_cen

def plot_metrics_separately(k_values, metrics_dict):
    """
    根据给定的k值和指标字典，为每个指标创建一个独立的子图进行绘制。
    """
    print("\n正在为每个评估指标生成独立的曲线图...")
    
    # 获取指标的数量，以确定需要创建多少个子图
    num_metrics = len(metrics_dict)
    
    # 创建一个包含 num_metrics 个子图的图表，它们共享X轴
    # figsize 的高度根据指标数量进行调整
    fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(12, 6 * num_metrics), sharex=True)
    
    if num_metrics == 1:
        axes = [axes]
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # 遍历每个指标和对应的子图轴
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[i]
        ax.plot(k_values, values, marker=markers[i], linestyle='-', color=colors[i], label=metric_name)
        
        # 为每个子图设置标题和Y轴标签
        ax.set_title(f'{metric_name} 随 k 值的变化曲线', fontsize=16)
        ax.set_ylabel(f'{metric_name} 得分', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

    # 为共享的X轴设置标签（只需要在最下面的子图上设置）
    axes[-1].set_xlabel('k 值 (近邻数)', fontsize=14)
    axes[-1].set_xticks(k_values) # 确保X轴刻度清晰

    # 添加一个总标题
    fig.suptitle('模型评估指标随 k 值的变化曲线 (基于LOOCV)', fontsize=20, y=1.0)
    
    # 自动调整布局以防止标题和标签重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

if __name__ == "__main__":
    DATA_FILE_PATH = 'semeion.data' 
    features, labels = load_and_process_data(DATA_FILE_PATH)
    
    if features is not None:
        N = len(features)
        
        print("\n--- 开始使用LOOCV评估不同k值下的所有模型指标 ---")
        
        # k_candidates = [3,7,11]
        k_candidates = [1, 3, 5, 7, 9, 11, 13, 15]
        
        all_metrics_results = {
            "准确率 (Accuracy)": [],
            "归一化互信息 (NMI)": [],
            "混淆熵 (CEN)": []
        }
        
        overall_start_time = time.time()
        
        for k in k_candidates:
            k_start_time = time.time()
            print(f"\n===== 正在为 k = {k} 进行评估... =====")

            all_true_labels = []
            all_predicted_labels = []
            
            model = KNN(k=k, verbose=False)

            for i in range(N):
                X_test_single = features[i]
                y_test_single = labels[i]
                X_train = np.delete(features, i, axis=0)
                y_train = np.delete(labels, i, axis=0)
                
                model.fit(X_train, y_train, verbose=False)
                prediction = model.predict_single(X_test_single)
                
                all_true_labels.append(y_test_single)
                all_predicted_labels.append(prediction)

                if (i + 1) % 400 == 0:
                    print(f"   ...已处理 {i + 1}/{N} 个样本")

            accuracy = metrics.accuracy_score(all_true_labels, all_predicted_labels)
            nmi_score = metrics.normalized_mutual_info_score(all_true_labels, all_predicted_labels)
            cen_score = calculate_cen(all_true_labels, all_predicted_labels)

            all_metrics_results["准确率 (Accuracy)"].append(accuracy)
            all_metrics_results["归一化互信息 (NMI)"].append(nmi_score)
            all_metrics_results["混淆熵 (CEN)"].append(cen_score)

            k_end_time = time.time()
            print(f"k = {k} 的评估完成 (耗时: {k_end_time - k_start_time:.2f} 秒)。")
            print(f"  -> 准确率: {accuracy:.4f}, NMI: {nmi_score:.4f}, CEN: {cen_score:.4f}")

        overall_end_time = time.time()
        print("\n\n================== 所有k值评估完成 ==================")
        
        accuracies = all_metrics_results["准确率 (Accuracy)"]
        best_idx = np.argmax(accuracies)
        best_k = k_candidates[best_idx]
        
        print(f"基于【准确率】，表现最好的 k 是: {best_k}")
        print("该k值下的各项指标为:")
        for metric_name, values in all_metrics_results.items():
            print(f"  - {metric_name}: {values[best_idx]:.4f}")
            
        print(f"\n总计运行时间: {(overall_end_time - overall_start_time)/60:.2f} 分钟")
        print("======================================================")
        
        
        if k_candidates:
            plot_metrics_separately(k_candidates, all_metrics_results)