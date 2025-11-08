import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import det, inv

# --- 解决中文显示、负号、字号问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 12         
plt.rcParams['axes.titlesize'] = 16    
plt.rcParams['axes.labelsize'] = 14    
plt.rcParams['legend.fontsize'] = 13   
# ------------------------------------


def manual_multivariate_normal_logpdf(X, mu, cov):
    """
    手动实现 scipy.stats.multivariate_normal.logpdf
    X: (N, d) 的数据点
    mu: (d,) 的均值
    cov: (d, d) 的协方差
    """
    N, d = X.shape
    
    # 1. 计算协方差矩阵的逆和行列式的对数
    inv_cov = inv(cov)
    log_det_cov = np.log(det(cov))
    
    # 2. 计算常数项
    log_const = -d/2 * np.log(2 * np.pi) - 0.5 * log_det_cov
    
    # 3. 计算马氏距离 (Mahalanobis distance) 的平方
    diff = X - mu
    
    term1 = diff @ inv_cov
    
    mahalanobis_sq = np.sum(term1 * diff, axis=1)
    
    # 4. 组合
    return log_const - 0.5 * mahalanobis_sq

def manual_pairwise_sqdist(A, B):
    """
    手动实现 scipy.spatial.distance.cdist(A, B, 'sqeuclidean')
    使用广播 (Broadcasting) 和矩阵乘法：(a-b)^2 = a^2 - 2ab + b^2
    A: (M, d)
    B: (N, d)
    """
    sum_A_sq = np.sum(A**2, axis=1, keepdims=True)
    
    sum_B_sq = np.sum(B**2, axis=1, keepdims=True)
    
    dot_prod = A @ B.T
    
    sq_dists = sum_A_sq - 2 * dot_prod + sum_B_sq.T
    
    return np.maximum(sq_dists, 0)

def manual_log_sum_exp(x, axis=1):
    """
    手动实现 scipy.special.logsumexp
    用于在对数空间中安全地计算 sum(exp(x))，防止下溢/上溢
    x: (M, N)
    """
    # 1. 找到每行的最大值
    max_val = np.max(x, axis=axis, keepdims=True) # (M, 1)
    
    # 2. 减去最大值后计算 exp 和 sum
    sum_exp = np.sum(np.exp(x - max_val), axis=axis)
    
    # 3. 加回最大值 (在对数空间)
    return np.squeeze(max_val, axis=axis) + np.log(sum_exp)

# ========================================================
# --- 1. 数据集参数设置 ---
# ========================================================
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
        true_label_idx = int(y_true[i] - 1) 
        pred_label_idx = int(y_pred[i] - 1)
        cm[true_label_idx, pred_label_idx] += 1
    return cm

def plot_confusion_matrix_manual(y_true, y_pred, title):
    """绘制混淆矩阵"""
    cm = manual_confusion_matrix(y_true, y_pred)
    acc = manual_accuracy(y_true, y_pred)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 5), yticklabels=range(1, 5))
    plt.title(f'{title}\nOverall Accuracy: {acc:.4f}')
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)

datasets = {}
for name, config in datasets_config.items():
    X, y = generate_dataset(config)
    datasets[name] = {'X': X, 'y': y, 'config': config}
    print(f"Generated {name} with {len(y)} samples.")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}

for ax, (name, data) in zip(axes, datasets.items()):
    X, y = data['X'], data['y']
    priors_str = str(data['config']['priors'])
    
    for class_label in range(1, 5):
        ax.scatter(X[y == class_label, 0], X[y == class_label, 1],
                   c=colors[class_label], label=f'Class {class_label}', 
                   s=20,          
                   alpha=0.6)
    
    ax.set_title(f"{data['config']['title']}\nPrior: {priors_str}")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

plt.tight_layout()

def gaussian_kernel_log_pdf_manual(X_test, X_train, h):
    """
    手动计算高斯核密度估计的对数似然 log(P(x))
    """
    N, d = X_train.shape
    
    # 1. cdist
    sq_dists = manual_pairwise_sqdist(X_test, X_train) # (M, N)
    
    # 2. 计算高斯核 K(u) 的对数
    log_kernel_vals = - (d/2) * np.log(2 * np.pi * h**2) - (sq_dists / (2 * h**2))
    
    # 3. 求和并取平均
    log_pdf = manual_log_sum_exp(log_kernel_vals, axis=1) - np.log(N)
    return log_pdf

class KDEClassifier:
    """手动实现KDE分类器"""
    def __init__(self, h=1.0):
        self.h = h
        self.X_train_ = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.X_train_[c] = X[y == c]
            if self.X_train_[c].shape[0] == 0:
                print(f"警告：类别 {c} 在此折 (fold) 中没有训练样本！")
        return self

    def predict_log_proba(self, X_test, priors, rule_type='MAP'):
        log_probs = []
        for c in self.classes_:
            X_train_c = self.X_train_[c]
            
            if X_train_c.shape[0] == 0:
                log_likelihood = np.full(X_test.shape[0], -np.inf)
            else:
                log_likelihood = gaussian_kernel_log_pdf_manual(X_test, X_train_c, self.h)
            
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

def run_kde_cv(X, y, priors, h_values):
    """为KDE手动运行5折交叉验证以优化h"""
    n_splits = 5
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) 
    fold_indices = np.array_split(indices, n_splits) 
    
    scores_map = {h: [] for h in h_values}
    scores_lrt = {h: [] for h in h_values}

    for h in h_values:
        print(f"  ... 正在测试带宽 h = {h}")
        for i in range(n_splits):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            kde = KDEClassifier(h=h)
            kde.fit(X_train, y_train)
            
            y_pred_map = kde.predict(X_val, priors, rule_type='MAP')
            scores_map[h].append(manual_accuracy(y_val, y_pred_map))
            
            y_pred_lrt = kde.predict(X_val, priors, rule_type='LRT')
            scores_lrt[h].append(manual_accuracy(y_val, y_pred_lrt))

    avg_scores_map = {h: np.mean(scores_map[h]) for h in h_values}
    avg_scores_lrt = {h: np.mean(scores_lrt[h]) for h in h_values}
    
    return avg_scores_map, avg_scores_lrt

# --- 运行实验 4.2 ---
bandwidth_range = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
results_kde_cv = {} 

fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True) 
fig.suptitle("KDE 带宽参数 (h) 对准确率的影响 (5折交叉验证)", fontsize=20)

for ax, (name, data) in zip(axes, datasets.items()):
    print(f"\n--- W (警告): 正在为 {data['config']['title']} 运行KDE交叉验证 ---")
    X, y = data['X'], data['y']
    priors = data['config']['priors']
    
    avg_scores_map, avg_scores_lrt = run_kde_cv(X, y, priors, bandwidth_range)
    
    results_kde_cv[name] = {'MAP': avg_scores_map, 'LRT': avg_scores_lrt}
    
    # --- 绘制图3 ---
    accs_lrt = [avg_scores_lrt[h] for h in bandwidth_range]
    accs_map = [avg_scores_map[h] for h in bandwidth_range]
    
    ax.plot(bandwidth_range, accs_lrt, 'o-', label='似然率测试 (LRT)')
    ax.plot(bandwidth_range, accs_map, 'o-', label='MAP 规则')
    
    best_h_map = max(avg_scores_map, key=avg_scores_map.get)
    best_acc_map = avg_scores_map[best_h_map]
    ax.plot(best_h_map, best_acc_map, 'bs', markersize=10, 
            label=f'最优 MAP (h={best_h_map:.1f})')
    
    best_h_lrt = max(avg_scores_lrt, key=avg_scores_lrt.get)
    best_acc_lrt = avg_scores_lrt[best_h_lrt]
    ax.plot(best_h_lrt, best_acc_lrt, 'r^', markersize=10, 
            label=f'最优 LRT (h={best_h_lrt:.1f})')

    ax.set_title(data['config']['title'])
    ax.set_xlabel('带宽参数 h')
    if name == 'A':
        ax.set_ylabel('交叉验证准确率')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

for dataset_name, results in results_kde_cv.items():
    print(f"\n--- {dataset_name} ---")
    lrt_scores = results['LRT']
    best_h_lrt = max(lrt_scores, key=lrt_scores.get)
    print(f"LRT: 最优 h = {best_h_lrt}, 准确率 = {lrt_scores[best_h_lrt]:.4f}")
    
    map_scores = results['MAP']
    best_h_map = max(map_scores, key=map_scores.get)
    print(f"MAP: 最优 h = {best_h_map}, 准确率 = {map_scores[best_h_map]:.4f}")


# --- 最后：显示所有绘制的图形 ---
print("\n正在显示所有图形窗口...")


# 从交叉验证结果中提取最优 h
best_h_values = {
    name: max(res['MAP'], key=res['MAP'].get)
    for name, res in results_kde_cv.items()
}
print("最优带宽参数 h：", best_h_values)

colors = ['r', 'b', 'g', 'm']  # 红、蓝、绿、紫

for name, data in datasets.items():
    X, y = data['X'], data['y']
    priors = data['config']['priors']
    h_best = best_h_values[name]

    print(f"\n正在绘制 Dataset {name} (h={h_best}) ...")

    # === 训练 KDE 分类器 ===
    kde = KDEClassifier(h=h_best)
    kde.fit(X, y)

    # === 网格范围 ===
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # ======================================================
    # === 每个数据集新建一个 Figure（独立显示）===
    # ======================================================
    fig = plt.figure(figsize=(12, 6))
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    for c in kde.classes_:
        X_c = kde.X_train_[c]
        log_dens_c = gaussian_kernel_log_pdf_manual(grid_points, X_c, h_best)
        dens_c = np.exp(log_dens_c).reshape(xx.shape)

        # 绘制等高线并返回 contour 对象
        contour = ax_left.contour(xx, yy, dens_c, 
                                colors=colors[c-1], 
                                alpha=0.7,
                                levels=8)  # 控制等高线层数（默认8层，可调整）
        
        # 添加每条等高线的数值标签
        ax_left.clabel(contour, 
                    inline=True,      # 标签放在线上
                    fontsize=8,       # 字号
                    fmt="%.3f")       # 数值格式（三位小数）

    ax_left.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax_left.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)


    ax_left.set_title(f"Dataset {name} - Density Contours\n(h={h_best})")
    ax_left.set_xlabel("Feature 1")
    ax_left.set_ylabel("Feature 2")
    ax_left.grid(True, linestyle='--', alpha=0.5)


    y_pred = kde.predict(X, priors, rule_type='MAP')
    acc = manual_accuracy(y, y_pred)

    for c in kde.classes_:
        mask_correct = (y == c) & (y_pred == c)
        mask_wrong = (y == c) & (y_pred != c)
        ax_right.scatter(X[mask_correct, 0], X[mask_correct, 1],
                         c=colors[c-1], marker='o', alpha=0.7, label=f"Class {c} Correct")
        ax_right.scatter(X[mask_wrong, 0], X[mask_wrong, 1],
                         c=colors[c-1], marker='x', alpha=0.7, label=f"Class {c} Wrong")

    ax_right.set_title(f"Dataset {name} - Classification Results\nAccuracy: {acc:.4f}")
    
    ax_right.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.show() 
