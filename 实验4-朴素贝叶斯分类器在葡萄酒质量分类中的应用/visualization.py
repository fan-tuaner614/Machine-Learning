# -*- coding: utf-8 -*-
"""
实验四：高级可视化 - 使用matplotlib绘制图表
包括：
1. 性能指标对比图
2. 混淆矩阵热力图
3. 特征重要性条形图
4. ROC曲线图
5. 特征选择效果对比图
"""

import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ==================== 导入基础分类器 ====================

class GaussianNaiveBayes:
    """手动实现的高斯朴素贝叶斯分类器"""
    
    def __init__(self):
        self.classes = []
        self.class_priors = {}
        self.feature_stats = {}
        
    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        n_features = len(X_train[0])
        
        class_samples = {}
        for i in range(n_samples):
            label = y_train[i]
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(X_train[i])
        
        self.classes = sorted(class_samples.keys())
        
        for c in self.classes:
            self.class_priors[c] = len(class_samples[c]) / n_samples
        
        for c in self.classes:
            self.feature_stats[c] = {}
            samples = class_samples[c]
            
            for feature_idx in range(n_features):
                feature_values = [sample[feature_idx] for sample in samples]
                mean = sum(feature_values) / len(feature_values)
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                std = math.sqrt(variance) if variance > 0 else 1e-6
                self.feature_stats[c][feature_idx] = (mean, std)
    
    def _gaussian_probability(self, x, mean, std):
        if std == 0:
            std = 1e-6
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    
    def _calculate_log_probability(self, sample, c):
        log_prob = math.log(self.class_priors[c])
        for feature_idx in range(len(sample)):
            mean, std = self.feature_stats[c][feature_idx]
            prob = self._gaussian_probability(sample[feature_idx], mean, std)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += -1e10
        return log_prob
    
    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            class_probs = {}
            for c in self.classes:
                class_probs[c] = self._calculate_log_probability(sample, c)
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return predictions
    
    def predict_proba(self, X_test):
        probabilities = []
        for sample in X_test:
            log_probs = {}
            for c in self.classes:
                log_probs[c] = self._calculate_log_probability(sample, c)
            max_log_prob = max(log_probs.values())
            exp_probs = {c: math.exp(log_probs[c] - max_log_prob) for c in self.classes}
            total = sum(exp_probs.values())
            probs = [exp_probs[c] / total for c in self.classes]
            probabilities.append(probs)
        return probabilities


# ==================== 数据读取 ====================

def read_csv_data(filename):
    """读取CSV格式的数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(',')
    feature_names = header[:-1]
    
    data = []
    labels = []
    for line in lines[1:]:
        row = line.strip().split(',')
        if row:
            data.append([float(x) for x in row[:-1]])
            labels.append(int(row[-1]))
    
    return data, labels, feature_names


# ==================== 性能评估函数 ====================

def calculate_metrics(y_true, y_pred, classes):
    """计算详细的性能指标"""
    metrics = {}
    
    for c in classes:
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == c and y_pred[i] == c)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] != c and y_pred[i] == c)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == c and y_pred[i] != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for label in y_true if label == c)
        
        metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return metrics


def calculate_roc_auc(y_true, y_proba, classes):
    """手动计算每个类别的ROC AUC"""
    auc_scores = {}
    
    for class_idx, c in enumerate(classes):
        binary_true = [1 if label == c else 0 for label in y_true]
        binary_proba = [probs[class_idx] for probs in y_proba]
        
        pairs = list(zip(binary_proba, binary_true))
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        n_pos = sum(binary_true)
        n_neg = len(binary_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            auc_scores[c] = 0.5
            continue
        
        fp = 0
        tp = 0
        fp_prev = 0
        tp_prev = 0
        auc = 0.0
        
        prev_score = None
        for score, label in pairs:
            if prev_score is not None and score != prev_score:
                auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
                fp_prev = fp
                tp_prev = tp
            
            if label == 1:
                tp += 1
            else:
                fp += 1
            prev_score = score
        
        auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
        auc = auc / (n_pos * n_neg)
        auc_scores[c] = auc
    
    return auc_scores


def calculate_roc_curve(y_true, y_proba, classes):
    """计算ROC曲线的FPR和TPR"""
    roc_data = {}
    
    for class_idx, c in enumerate(classes):
        binary_true = [1 if label == c else 0 for label in y_true]
        binary_proba = [probs[class_idx] for probs in y_proba]
        
        pairs = list(zip(binary_proba, binary_true))
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        n_pos = sum(binary_true)
        n_neg = len(binary_true) - n_pos
        
        fpr_list = [0.0]
        tpr_list = [0.0]
        
        fp = 0
        tp = 0
        
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            fpr = fp / n_neg if n_neg > 0 else 0
            tpr = tp / n_pos if n_pos > 0 else 0
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        
        roc_data[c] = (fpr_list, tpr_list)
    
    return roc_data


# ==================== 特征重要性分析 ====================

def calculate_feature_importance_anova(X_train, y_train, feature_names):
    """使用ANOVA F值计算特征重要性"""
    n_features = len(X_train[0])
    f_scores = []
    
    class_groups = defaultdict(list)
    for i in range(len(X_train)):
        class_groups[y_train[i]].append(X_train[i])
    
    classes = sorted(class_groups.keys())
    
    for feature_idx in range(n_features):
        all_values = [X_train[i][feature_idx] for i in range(len(X_train))]
        grand_mean = sum(all_values) / len(all_values)
        
        ssb = 0.0
        for c in classes:
            group_values = [sample[feature_idx] for sample in class_groups[c]]
            group_mean = sum(group_values) / len(group_values)
            ssb += len(group_values) * (group_mean - grand_mean) ** 2
        
        ssw = 0.0
        for c in classes:
            group_values = [sample[feature_idx] for sample in class_groups[c]]
            group_mean = sum(group_values) / len(group_values)
            ssw += sum((val - group_mean) ** 2 for val in group_values)
        
        df_between = len(classes) - 1
        df_within = len(X_train) - len(classes)
        
        msb = ssb / df_between if df_between > 0 else 0
        msw = ssw / df_within if df_within > 0 else 1e-10
        f_value = msb / msw if msw > 0 else 0
        
        f_scores.append(f_value)
    
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = f_scores[i]
    
    return importance


def select_top_features(X, feature_names, importance, k=5):
    """选择前k个最重要的特征"""
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_k_features = [name for name, score in sorted_features[:k]]
    
    feature_indices = [feature_names.index(name) for name in top_k_features]
    
    X_selected = []
    for sample in X:
        selected_sample = [sample[idx] for idx in feature_indices]
        X_selected.append(selected_sample)
    
    return X_selected, top_k_features, feature_indices


# ==================== 可视化函数 ====================

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    class_names = ['低质量(0)', '中等质量(1)', '高质量(2)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='朴素贝叶斯分类器 - 混淆矩阵',
           ylabel='真实类别',
           xlabel='预测类别')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵热力图已保存至: {save_path}")
    plt.close()


def plot_feature_importance(importance, save_path='feature_importance.png'):
    """绘制特征重要性条形图"""
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, scores, color='steelblue', alpha=0.8)
    
    # 为条形添加渐变色
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
    for bar, color in zip(bars, colors[::-1]):
        bar.set_color(color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('ANOVA F-值', fontsize=12, fontweight='bold')
    ax.set_title('特征重要性排序 (基于方差分析)', fontsize=14, fontweight='bold')
    
    # 在条形末端添加数值
    for i, score in enumerate(scores):
        ax.text(score + 2, i, f'{score:.2f}', 
                va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存至: {save_path}")
    plt.close()


def plot_performance_metrics(metrics, auc_scores, classes, save_path='performance_metrics.png'):
    """绘制性能指标对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    class_names = ['低质量', '中等质量', '高质量']
    
    # 子图1: 精确率、召回率、F1对比
    ax1 = axes[0, 0]
    x = np.arange(len(classes))
    width = 0.25
    
    precision = [metrics[c]['precision'] for c in classes]
    recall = [metrics[c]['recall'] for c in classes]
    f1 = [metrics[c]['f1'] for c in classes]
    
    ax1.bar(x - width, precision, width, label='精确率', color='#3498db')
    ax1.bar(x, recall, width, label='召回率', color='#2ecc71')
    ax1.bar(x + width, f1, width, label='F1分数', color='#e74c3c')
    
    ax1.set_xlabel('类别', fontweight='bold')
    ax1.set_ylabel('分数', fontweight='bold')
    ax1.set_title('各类别性能指标对比', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 子图2: AUC值对比
    ax2 = axes[0, 1]
    auc_values = [auc_scores[c] for c in classes]
    bars = ax2.bar(class_names, auc_values, color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.8)
    
    ax2.set_xlabel('类别', fontweight='bold')
    ax2.set_ylabel('AUC值', fontweight='bold')
    ax2.set_title('各类别AUC值对比', fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在条形上添加数值
    for bar, value in zip(bars, auc_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: 支持度分布
    ax3 = axes[1, 0]
    support = [metrics[c]['support'] for c in classes]
    colors_pie = ['#e74c3c', '#3498db', '#2ecc71']
    wedges, texts, autotexts = ax3.pie(support, labels=class_names, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax3.set_title('测试集类别分布', fontweight='bold')
    
    # 子图4: 综合性能雷达图
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 创建表格显示整体指标
    table_data = []
    table_data.append(['指标', '低质量', '中等质量', '高质量'])
    table_data.append(['精确率', f"{precision[0]:.2f}", f"{precision[1]:.2f}", f"{precision[2]:.2f}"])
    table_data.append(['召回率', f"{recall[0]:.2f}", f"{recall[1]:.2f}", f"{recall[2]:.2f}"])
    table_data.append(['F1分数', f"{f1[0]:.2f}", f"{f1[1]:.2f}", f"{f1[2]:.2f}"])
    table_data.append(['AUC', f"{auc_values[0]:.3f}", f"{auc_values[1]:.3f}", f"{auc_values[2]:.3f}"])
    table_data.append(['支持度', f"{support[0]}", f"{support[1]}", f"{support[2]}"])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行标题样式
    for i in range(1, 6):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')
    
    ax4.set_title('详细性能指标表', fontweight='bold', pad=20)
    
    fig.suptitle('朴素贝叶斯分类器 - 综合性能分析', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"性能指标图已保存至: {save_path}")
    plt.close()


def plot_roc_curves(roc_data, auc_scores, classes, save_path='roc_curves.png'):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    class_names = ['低质量(0)', '中等质量(1)', '高质量(2)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, c in enumerate(classes):
        fpr, tpr = roc_data[c]
        auc = auc_scores[c]
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('ROC曲线 - 各类别', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC曲线图已保存至: {save_path}")
    plt.close()


def plot_feature_selection_comparison(results, save_path='feature_selection.png'):
    """绘制特征选择效果对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    k_values = [r['k'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    train_times = [r['train_time'] * 1000 for r in results]  # 转换为毫秒
    
    # 子图1: 准确率随特征数量变化
    ax1 = axes[0]
    line1 = ax1.plot(k_values, accuracies, marker='o', markersize=8, 
                     linewidth=2, color='#2ecc71', label='准确率')
    ax1.set_xlabel('特征数量', fontsize=12, fontweight='bold')
    ax1.set_ylabel('准确率', fontsize=12, fontweight='bold', color='#2ecc71')
    ax1.set_title('特征数量对模型性能的影响', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # 标注最大值
    max_idx = accuracies.index(max(accuracies))
    ax1.annotate(f'最佳: {accuracies[max_idx]:.4f}\n特征数: {k_values[max_idx]}',
                xy=(k_values[max_idx], accuracies[max_idx]),
                xytext=(k_values[max_idx] + 1, accuracies[max_idx] - 0.02),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    # 添加第二个y轴显示训练时间
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(k_values, train_times, marker='s', markersize=8,
                          linewidth=2, color='#e74c3c', linestyle='--', label='训练时间')
    ax1_twin.set_ylabel('训练时间 (毫秒)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 子图2: 准确率提升对比
    ax2 = axes[1]
    baseline_accuracy = accuracies[-1]  # 使用所有特征的准确率作为基准
    improvements = [(acc - baseline_accuracy) * 100 for acc in accuracies]
    
    colors_bar = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]
    bars = ax2.bar(k_values, improvements, color=colors_bar, alpha=0.8, edgecolor='black')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('特征数量', fontsize=12, fontweight='bold')
    ax2.set_ylabel('准确率提升 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('相对于全特征集的准确率提升', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在条形上添加数值
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%', ha='center', 
                va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=9)
    
    fig.suptitle('特征选择方法优化效果分析', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征选择对比图已保存至: {save_path}")
    plt.close()


# ==================== 主程序 ====================

def main():
    print("=" * 100)
    print("实验四：高级可视化 - 使用matplotlib绘制图表")
    print("=" * 100)
    print()
    
    # 读取数据
    print("步骤1: 读取数据...")
    X_train, y_train, feature_names = read_csv_data('train_set.csv')
    X_test, y_test, _ = read_csv_data('test_set.csv')
    print(f"✓ 训练集: {len(X_train)} 样本")
    print(f"✓ 测试集: {len(X_test)} 样本")
    print()
    
    # 训练基础模型
    print("步骤2: 训练朴素贝叶斯分类器...")
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)
    classes = nb.classes
    
    accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / len(y_test)
    print(f"✓ 训练完成！准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # 计算性能指标
    print("步骤3: 计算性能指标...")
    metrics = calculate_metrics(y_test, y_pred, classes)
    auc_scores = calculate_roc_auc(y_test, y_proba, classes)
    roc_data = calculate_roc_curve(y_test, y_proba, classes)
    print("✓ 性能指标计算完成")
    print()
    
    # 计算混淆矩阵
    print("步骤4: 生成混淆矩阵...")
    cm = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
    for i in range(len(y_test)):
        true_idx = classes.index(y_test[i])
        pred_idx = classes.index(y_pred[i])
        cm[true_idx][pred_idx] += 1
    
    cm_array = np.array(cm)
    plot_confusion_matrix(cm_array, classes)
    print()
    
    # 绘制性能指标图
    print("步骤5: 绘制性能指标图...")
    plot_performance_metrics(metrics, auc_scores, classes)
    print()
    
    # 绘制ROC曲线
    print("步骤6: 绘制ROC曲线...")
    plot_roc_curves(roc_data, auc_scores, classes)
    print()
    
    # 特征重要性分析
    print("步骤7: 特征重要性分析...")
    importance = calculate_feature_importance_anova(X_train, y_train, feature_names)
    plot_feature_importance(importance)
    print()
    
    # 特征选择优化
    print("步骤8: 特征选择优化...")
    results = []
    
    for k in [3, 5, 7, 9, 11]:
        X_train_selected, selected_features, _ = select_top_features(
            X_train, feature_names, importance, k
        )
        X_test_selected, _, _ = select_top_features(
            X_test, feature_names, importance, k
        )
        
        start_time = time.time()
        nb_opt = GaussianNaiveBayes()
        nb_opt.fit(X_train_selected, y_train)
        train_time = time.time() - start_time
        
        y_pred_opt = nb_opt.predict(X_test_selected)
        accuracy_opt = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred_opt[i]) / len(y_test)
        
        results.append({
            'k': k,
            'features': selected_features,
            'accuracy': accuracy_opt,
            'train_time': train_time
        })
        
        print(f"  特征数={k}, 准确率={accuracy_opt:.4f}")
    
    plot_feature_selection_comparison(results)
    print()
    
    # 总结
    print("=" * 100)
    print("可视化完成！生成的图表:")
    print("=" * 100)
    print("1. confusion_matrix.png - 混淆矩阵热力图")
    print("2. performance_metrics.png - 综合性能指标图")
    print("3. roc_curves.png - ROC曲线图")
    print("4. feature_importance.png - 特征重要性条形图")
    print("5. feature_selection.png - 特征选择效果对比图")
    print("=" * 100)
    print()
    
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"最佳配置: 使用 {best_result['k']} 个特征，准确率 {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"选择的特征: {', '.join(best_result['features'])}")
    print()


if __name__ == "__main__":
    main()
