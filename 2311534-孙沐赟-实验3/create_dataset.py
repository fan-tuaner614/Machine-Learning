import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为"黑体"
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size'] = 12         # 设置基础字号
plt.rcParams['axes.titlesize'] = 16    # 调大标题字号
plt.rcParams['axes.labelsize'] = 14    # 调大坐标轴标签字号
plt.rcParams['legend.fontsize'] = 13   # 调大图例字号

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