import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为"黑体"
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size'] = 12         # 设置基础字号
plt.rcParams['axes.titlesize'] = 16    # 调大标题字号
plt.rcParams['axes.labelsize'] = 14    # 调大坐标轴标签字号
plt.rcParams['legend.fontsize'] = 13   # 调大图例字号

# --- 1. 数据集参数设置---
means = {
    1: np.array([1, 2]),
    2: np.array([3, 5]),
    3: np.array([6, 3]),
    4: np.array([4, 7])
}
covs = {
    1: np.array([[1.5, 0], [0, 1.5]]),
    2: np.array([[1.5, 0], [0, 1.5]]),
    3: np.array([[1.5, 0], [0, 1.5]]),
    4: np.array([[1.5, 0], [0, 1.5]])
}

# --- 先验概率设置---
datasets_config = {
    'A': {'priors': [0.25, 0.25, 0.25, 0.25], 'N': 1200, 'title': 'Dataset A (均衡先验)'},
    'B': {'priors': [0.5, 0.2, 0.2, 0.1], 'N': 1200, 'title': 'Dataset B (偏斜先验1)'},
    'C': {'priors': [0.1, 0.1, 0.3, 0.5], 'N': 1200, 'title': 'Dataset C (偏斜先验2)'}
}

# --- 2. 数据集生成函数 ---
def generate_dataset(config):
    """根据配置生成数据集"""
    X = []
    y = []
    for i, prior in enumerate(config['priors']):
        n_samples = int(config['N'] * prior)
        class_label = i + 1
        X.append(np.random.multivariate_normal(means[class_label], covs[class_label], n_samples))
        y.append(np.full(n_samples, class_label))
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    # 打乱数据
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y

# --- 3. 手动实现评估工具 ---
def manual_accuracy(y_true, y_pred):
    """手动计算准确率"""
    return np.mean(y_true == y_pred)

def manual_confusion_matrix(y_true, y_pred, n_classes=4):
    """手动计算混淆矩阵"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        true_label_idx = int(y_true[i] - 1) # 类别 1-4 对应索引 0-3
        pred_label_idx = int(y_pred[i] - 1)
        cm[true_label_idx, pred_label_idx] += 1
    return cm

def manual_pairwise_sqdist(A, B):
    """
    手动实现 scipy.spatial.distance.cdist(A, B, 'sqeuclidean')
    使用广播 (Broadcasting) 和矩阵乘法：(a-b)^2 = a^2 - 2ab + b^2
    A: (M, d)
    B: (N, d)
    """
    # a^2, (M, 1)
    sum_A_sq = np.sum(A**2, axis=1, keepdims=True)
    
    # b^2, (N, 1)
    sum_B_sq = np.sum(B**2, axis=1, keepdims=True)
    
    # 2ab, (M, N)
    dot_prod = A @ B.T
    
    # (M, 1) - 2*(M, N) + (1, N) -> (M, N)
    sq_dists = sum_A_sq - 2 * dot_prod + sum_B_sq.T
    
    # 可能有极小的负值（由于浮点数精度），取0
    return np.maximum(sq_dists, 0)

def plot_confusion_matrix_manual(y_true, y_pred, title):
    """绘制混淆矩阵"""
    cm = manual_confusion_matrix(y_true, y_pred)
    acc = manual_accuracy(y_true, y_pred)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 5), yticklabels=range(1, 5))
    plt.title(f'{title}\nOverall Accuracy: {acc:.4f}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

# --- 4. 生成数据并可视化---
datasets = {}
for name, config in datasets_config.items():
    X, y = generate_dataset(config)
    datasets[name] = {'X': X, 'y': y, 'config': config}
    print(f"Generated {name} with {len(y)} samples.")

# --- 3. 可视化---
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}

for ax, (name, data) in zip(axes, datasets.items()):
    X, y = data['X'], data['y']
    priors_str = str(data['config']['priors'])
    
    for class_label in range(1, 5):
        ax.scatter(X[y == class_label, 0], X[y == class_label, 1],
                   c=colors[class_label], label=f'Class {class_label}', s=25, alpha=0.5)
    
    ax.set_title(f"{data['config']['title']}\nPrior: {priors_str}")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

plt.tight_layout()
plt.show()

class KNNDensityClassifier:
    """
    手动实现 k-NN 密度估计分类器
    注意：这不是 k-NN 分类器，而是 k-NN 密度估计 + 贝叶斯决策
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train_ = {} # 按类别存储训练数据
        self.N_i_ = {}     # 按类别存储样本数
        self.classes_ = []

    def fit(self, X, y):
        """
        'fit' 在这里是按类别把训练数据全背下来
        """
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.X_train_[c] = X[y == c]
            self.N_i_[c] = self.X_train_[c].shape[0]
            
            # 关键的健全性检查
            if self.k > self.N_i_[c]:
                print(f"警告：k={self.k} 大于类别 {c} 的样本数 {self.N_i_[c]}！")
                # 在这种情况下，我们被迫使用最大可用k值
                # 但在本次实验的全数据集上不应发生
                pass 
        return self

    def predict_log_proba(self, X_test, priors, rule_type='MAP'):
        log_probs = []
        
        for c in self.classes_:
            X_train_c = self.X_train_[c]
            N_i = self.N_i_[c]
            
            if N_i == 0 or self.k > N_i:
                 log_likelihood = np.full(X_test.shape[0], -np.inf)
            else:
                # 1. 计算 X_test 到该类所有训练点的平方距离
                sq_dists = manual_pairwise_sqdist(X_test, X_train_c) # (M, N_i)
                
                # 2. 转换成真实距离
                dists = np.sqrt(sq_dists)
                
                # 3. 沿行排序 (对每个测试点)，找到 k 个最近的邻居
                dists.sort(axis=1)
                
                # 4. 找出第 k 个邻居的距离 (索引是 k-1)
                d_ik = dists[:, self.k - 1] # (M,)
                
                # 5. 处理 d_ik == 0 的情况 (测试点和训练点重合)
                #    给一个极小值防止 log(0)
                d_ik[d_ik == 0] = 1e-9 
                
                # 6. 计算 log(p(x|wi)) = log(k) - log(Ni) - log(Vi)
                #    log(Vi) = log(pi * d_ik^2) = log(pi) + 2*log(d_ik)
                log_Vi = np.log(np.pi) + 2 * np.log(d_ik)
                
                log_likelihood = np.log(self.k) - np.log(N_i) - log_Vi
            
            # 7. 应用决策规则
            if rule_type == 'MAP':
                prior_prob = priors[int(c)-1]
                if prior_prob == 0:
                    log_prior = -np.inf
                else:
                    log_prior = np.log(prior_prob)
                log_posterior = log_likelihood + log_prior
                log_probs.append(log_posterior)
            else: # LRT
                log_probs.append(log_likelihood)
                
        return np.stack(log_probs, axis=1)

    def predict(self, X_test, priors, rule_type='MAP'):
        log_posteriors = self.predict_log_proba(X_test, priors, rule_type)
        pred_indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[pred_indices]

k_range = [1, 3, 5, 8, 10] # 
results_knn = {} 

fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
fig.suptitle("k-NN 密度估计的 k 值对准确率的影响", fontsize=20)

for ax, (name, data) in zip(axes, datasets.items()):
    print(f"\n--- 正在为 {data['config']['title']} 运行 k-NN 密度估计 ---")
    X, y = data['X'], data['y']
    priors = data['config']['priors']
    
    accs_map = []
    accs_lrt = []
    
    table_row_map = {}
    table_row_lrt = {}

    for k in k_range:
        print(f"  ... 正在测试 k = {k}")
        knn_clf = KNNDensityClassifier(k=k)
        knn_clf.fit(X, y)
        
        # 评估 MAP
        y_pred_map = knn_clf.predict(X, priors, rule_type='MAP')
        acc_map = manual_accuracy(y, y_pred_map)
        accs_map.append(acc_map)
        table_row_map[k] = acc_map
        
        # 评估 LRT
        y_pred_lrt = knn_clf.predict(X, priors, rule_type='LRT')
        acc_lrt = manual_accuracy(y, y_pred_lrt)
        accs_lrt.append(acc_lrt)
        table_row_lrt[k] = acc_lrt
        
    results_knn[name] = {'MAP': table_row_map, 'LRT': table_row_lrt}
        
    # --- 绘制图5 ---
    ax.plot(k_range, accs_lrt, 'o-', label='似然率测试 (LRT)')
    ax.plot(k_range, accs_map, 'o-', label='MAP 规则')
    
    # 找到并标记最优k
    best_k_map = k_range[np.argmax(accs_map)]
    best_acc_map = np.max(accs_map)
    ax.plot(best_k_map, best_acc_map, 'bs', markersize=10, 
            label=f'最优 MAP (k={best_k_map})')
    
    best_k_lrt = k_range[np.argmax(accs_lrt)]
    best_acc_lrt = np.max(accs_lrt)
    ax.plot(best_k_lrt, best_acc_lrt, 'r^', markersize=10, 
            label=f'最优 LRT (k={best_k_lrt})')

    ax.set_title(data['config']['title'])
    ax.set_xlabel('k 值')
    if name == 'A':
        ax.set_ylabel('分类准确率')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

for dataset_name, results in results_knn.items():
    print(f"\n--- {dataset_name} ---")
    print(f"LRT: {results['LRT']}")
    print(f"MAP: {results['MAP']}")

# --- 最后：显示所有绘制的图形 ---
print("\n正在显示所有图形窗口...")
plt.show()