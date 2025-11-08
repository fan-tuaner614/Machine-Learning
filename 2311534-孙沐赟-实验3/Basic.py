import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# --- 解决中文显示、负号、字号问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为"黑体"
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size'] = 12         # 设置基础字号
plt.rcParams['axes.titlesize'] = 16    # 调大标题字号
plt.rcParams['axes.labelsize'] = 14    # 调大坐标轴标签字号
plt.rcParams['legend.fontsize'] = 13   # 调大图例字号
# ------------------------------------

# --- 1. 数据集参数设置 ---
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
        true_label_idx = int(y_true[i] - 1) # 类别 1-4 对应索引 0-3
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

# --- 4. 生成数据---
datasets = {}
for name, config in datasets_config.items():
    X, y = generate_dataset(config)
    datasets[name] = {'X': X, 'y': y, 'config': config}
    print(f"Generated {name} with {len(y)} samples.")


# --- 5. 可视化---
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}

for ax, (name, data) in zip(axes, datasets.items()):
    X, y = data['X'], data['y']
    priors_str = str(data['config']['priors'])
    
    for class_label in range(1, 5):
        # 使用了更新后的大小和透明度
        ax.scatter(X[y == class_label, 0], X[y == class_label, 1],
                   c=colors[class_label], label=f'Class {class_label}', 
                   s=20,          
                   alpha=0.6)
    
    ax.set_title(f"{data['config']['title']}\nPrior: {priors_str}")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

plt.tight_layout()

# ========================================================
# --- 实验 4.1: 参数估计分类器
# ========================================================

print("\n--- 实验 4.1: 开始执行参数估计分类器 ---")

class ParametricClassifier:
    """
    参数估计分类器。
    我们“已知”真实的均值 (means) 和协方差 (covs)。
    """
    def __init__(self, means, covs):
        self.means = means
        self.covs = covs
        self.classes_ = list(means.keys()) # 类别 [1, 2, 3, 4]
        
        # 预先创建 PDF 对象，用于计算 P(x|w_i)
        # 这是参数估计的核心：我们用已知的均值和协方差来构建模型
        self.pdfs_ = {c: multivariate_normal(mean=means[c], cov=covs[c]) for c in self.classes_}

    def predict_log_proba(self, X, priors, rule_type='MAP'):
        """
        计算对数后验概率或对数似然。
        priors: 列表, e.g., [0.25, 0.25, 0.25, 0.25]
        rule_type: 'MAP' 或 'LRT'
        """
        log_probs = []
        for c in self.classes_:
            # P(x|wi)
            # 使用 logpdf 来防止数值下溢，速度也更快
            log_likelihood = self.pdfs_[c].logpdf(X) 
            
            if rule_type == 'MAP':
                # MAP 规则: log(P(x|wi)) + log(P(wi))
                prior_prob = priors[c-1] # 索引 0-3 对应类别 1-4
                
                # 处理先验为0的极端情况
                if prior_prob == 0:
                    log_prior = -np.inf
                else:
                    log_prior = np.log(prior_prob)
                    
                log_posterior = log_likelihood + log_prior
                log_probs.append(log_posterior)
            else: # LRT
                # 似然率测试: log(P(x|wi))
                log_probs.append(log_likelihood)
                
        # (N_samples, N_classes)
        return np.stack(log_probs, axis=1)

    def predict(self, X, priors, rule_type='MAP'):
        """进行分类"""
        log_posteriors = self.predict_log_proba(X, priors, rule_type)
        
        # 找到最大概率的索引 (0-3)，然后映射回类别 (1-4)
        pred_indices = np.argmax(log_posteriors, axis=1)
        # self.classes_ 是 [1, 2, 3, 4]
        return np.array(self.classes_)[pred_indices]

results_parametric = {}
classifier = ParametricClassifier(means, covs)

# 循环 A, B, C 三个数据集
for name, data in datasets.items():
    X, y_true = data['X'], data['y']
    priors = data['config']['priors']
    title = data['config']['title']
    
    print(f"\n--- 正在处理 {title} (生成图2) ---")
    
    # 1. 运行似然率测试 (LRT)
    y_pred_lrt = classifier.predict(X, priors, rule_type='LRT')
    acc_lrt = manual_accuracy(y_true, y_pred_lrt)
    results_parametric[f'{name}_LRT_Acc'] = acc_lrt
    
    plot_confusion_matrix_manual(y_true, y_pred_lrt, f'{title} - 似然率测试 (LRT)')
    
    # 2. 运行最大后验概率 (MAP)
    y_pred_map = classifier.predict(X, priors, rule_type='MAP')
    acc_map = manual_accuracy(y_true, y_pred_map)
    results_parametric[f'{name}_MAP_Acc'] = acc_map
    
    plot_confusion_matrix_manual(y_true, y_pred_map, f'{title} - MAP 规则')

for key, val in results_parametric.items():
    print(f"{key}: {val:.4f}")

# --- 最后：显示所有绘制的图形 ---
print("\n正在显示所有图形窗口...")
plt.show()