# -*- coding: utf-8 -*-
"""
实验四：高级要求 - 多分类器对比
包括：
1. 朴素贝叶斯分类器（手动实现）
2. 逻辑回归（sklearn）
3. 随机森林（sklearn）
4. 支持向量机（sklearn）
5. 多分类器性能对比表格和可视化
6. 多模型ROC曲线对比
7. 各类别详细性能对比
"""

import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 高斯朴素贝叶斯分类器 ====================

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
        return np.array(probabilities)


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

def calculate_detailed_metrics(y_true, y_pred, y_proba, classes):
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
    
    # 计算AUC（使用sklearn的roc_curve）
    y_true_bin = label_binarize(y_true, classes=classes)
    
    auc_scores = {}
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_scores[c] = auc(fpr, tpr)
        metrics[c]['auc'] = auc_scores[c]
    
    # 计算宏平均和加权平均
    total_support = sum(metrics[c]['support'] for c in classes)
    
    macro_precision = sum(metrics[c]['precision'] for c in classes) / len(classes)
    macro_recall = sum(metrics[c]['recall'] for c in classes) / len(classes)
    macro_f1 = sum(metrics[c]['f1'] for c in classes) / len(classes)
    macro_auc = sum(metrics[c]['auc'] for c in classes) / len(classes)
    
    weighted_precision = sum(metrics[c]['precision'] * metrics[c]['support'] for c in classes) / total_support
    weighted_recall = sum(metrics[c]['recall'] * metrics[c]['support'] for c in classes) / total_support
    weighted_f1 = sum(metrics[c]['f1'] * metrics[c]['support'] for c in classes) / total_support
    weighted_auc = sum(metrics[c]['auc'] * metrics[c]['support'] for c in classes) / total_support
    
    return metrics, macro_precision, macro_recall, macro_f1, macro_auc, weighted_precision, weighted_recall, weighted_f1, weighted_auc


# ==================== 训练所有分类器 ====================

def train_all_classifiers(X_train, y_train, X_test, y_test, classes):
    """训练所有分类器并记录性能"""
    results = {}
    
    print("=" * 100)
    print("训练所有分类器...")
    print("=" * 100)
    print()
    
    # 1. 朴素贝叶斯
    print("1. 训练朴素贝叶斯分类器...")
    start_time = time.time()
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_nb = nb.predict(X_test)
    y_proba_nb = nb.predict_proba(X_test)
    accuracy_nb = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred_nb[i]) / len(y_test)
    
    metrics_nb, macro_p, macro_r, macro_f1, macro_auc, weighted_p, weighted_r, weighted_f1, weighted_auc = \
        calculate_detailed_metrics(y_test, y_pred_nb, y_proba_nb, classes)
    
    results['朴素贝叶斯'] = {
        'model': nb,
        'y_pred': y_pred_nb,
        'y_proba': y_proba_nb,
        'accuracy': accuracy_nb,
        'train_time': train_time,
        'metrics': metrics_nb,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_auc': macro_auc
    }
    print(f"   准确率: {accuracy_nb:.4f}, 训练时间: {train_time:.6f}秒")
    print()
    
    # 2. 逻辑回归
    print("2. 训练逻辑回归分类器...")
    start_time = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)
    accuracy_lr = lr.score(X_test, y_test)
    
    metrics_lr, macro_p, macro_r, macro_f1, macro_auc, weighted_p, weighted_r, weighted_f1, weighted_auc = \
        calculate_detailed_metrics(y_test, y_pred_lr, y_proba_lr, classes)
    
    results['逻辑回归'] = {
        'model': lr,
        'y_pred': y_pred_lr,
        'y_proba': y_proba_lr,
        'accuracy': accuracy_lr,
        'train_time': train_time,
        'metrics': metrics_lr,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_auc': macro_auc
    }
    print(f"   准确率: {accuracy_lr:.4f}, 训练时间: {train_time:.6f}秒")
    print()
    
    # 3. 随机森林
    print("3. 训练随机森林分类器...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    accuracy_rf = rf.score(X_test, y_test)
    
    metrics_rf, macro_p, macro_r, macro_f1, macro_auc, weighted_p, weighted_r, weighted_f1, weighted_auc = \
        calculate_detailed_metrics(y_test, y_pred_rf, y_proba_rf, classes)
    
    results['随机森林'] = {
        'model': rf,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf,
        'accuracy': accuracy_rf,
        'train_time': train_time,
        'metrics': metrics_rf,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_auc': macro_auc
    }
    print(f"   准确率: {accuracy_rf:.4f}, 训练时间: {train_time:.6f}秒")
    print()
    
    # 4. 支持向量机
    print("4. 训练支持向量机分类器...")
    start_time = time.time()
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred_svm = svm.predict(X_test)
    y_proba_svm = svm.predict_proba(X_test)
    accuracy_svm = svm.score(X_test, y_test)
    
    metrics_svm, macro_p, macro_r, macro_f1, macro_auc, weighted_p, weighted_r, weighted_f1, weighted_auc = \
        calculate_detailed_metrics(y_test, y_pred_svm, y_proba_svm, classes)
    
    results['支持向量机'] = {
        'model': svm,
        'y_pred': y_pred_svm,
        'y_proba': y_proba_svm,
        'accuracy': accuracy_svm,
        'train_time': train_time,
        'metrics': metrics_svm,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_auc': macro_auc
    }
    print(f"   准确率: {accuracy_svm:.4f}, 训练时间: {train_time:.6f}秒")
    print()
    
    return results


# ==================== 表格输出 ====================

def print_comparison_table(results, classes):
    """打印表6: 不同分类器在葡萄酒质量分类任务上的性能比较"""
    print("\n" + "=" * 100)
    print("表6: 不同分类器在葡萄酒质量分类任务上的性能比较")
    print("=" * 100)
    print()
    
    print(f"{'分类器':<15} {'准确率':>12} {'宏平均 F1':>15} {'加权平均 F1':>15} {'宏观 AUC':>15}")
    print("-" * 100)
    
    for name in ['朴素贝叶斯', '逻辑回归', '随机森林', '支持向量机']:
        r = results[name]
        print(f"{name:<15} {r['accuracy']:>12.4f} {r['macro_f1']:>15.2f} "
              f"{r['weighted_f1']:>15.2f} {r['macro_auc']:>15.3f}")
    
    print()


def print_per_class_performance(results, classes):
    """打印各类别详细性能"""
    print("\n" + "=" * 100)
    print("各类别详细性能对比")
    print("=" * 100)
    
    class_names = ['低质量(0)', '中等质量(1)', '高质量(2)']
    
    for i, c in enumerate(classes):
        print()
        print(f"\n类别: {class_names[i]}")
        print("-" * 100)
        print(f"{'分类器':<15} {'精确率':>12} {'召回率':>12} {'F1分数':>12} {'AUC':>12}")
        print("-" * 100)
        
        for name in ['朴素贝叶斯', '逻辑回归', '随机森林', '支持向量机']:
            m = results[name]['metrics'][c]
            print(f"{name:<15} {m['precision']:>12.4f} {m['recall']:>12.4f} "
                  f"{m['f1']:>12.4f} {m['auc']:>12.4f}")


# ==================== 可视化函数 ====================

def plot_multi_model_roc_curves(results, y_test, classes, save_path='multi_model_roc.png'):
    """绘制多模型ROC曲线对比（宏观平均）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'朴素贝叶斯': '#e74c3c', '逻辑回归': '#3498db', 
              '随机森林': '#2ecc71', '支持向量机': '#f39c12'}
    
    y_true_bin = label_binarize(y_test, classes=classes)
    
    for name in ['朴素贝叶斯', '逻辑回归', '随机森林', '支持向量机']:
        y_proba = results[name]['y_proba']
        
        # 计算宏观平均ROC
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= len(classes)
        mean_auc = results[name]['macro_auc']
        
        ax.plot(all_fpr, mean_tpr, color=colors[name], lw=2,
                label=f'{name} (AUC = {mean_auc:.3f})')
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('多模型 ROC 曲线比较（宏观平均 ROC 曲线上的表现）', fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"多模型ROC曲线图已保存至: {save_path}")
    plt.close()


def plot_per_class_roc_curves(results, y_test, classes, save_path='per_class_roc.png'):
    """绘制图4: 概率校准曲线（各类别ROC曲线）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    class_names = ['低质量类别曲线', '中等质量类别曲线', '高质量类别曲线']
    colors = {'朴素贝叶斯': '#e74c3c', '逻辑回归': '#3498db', 
              '随机森林': '#2ecc71', '支持向量机': '#f39c12'}
    
    y_true_bin = label_binarize(y_test, classes=classes)
    
    for idx, (ax, c, class_name) in enumerate(zip(axes, classes, class_names)):
        for name in ['朴素贝叶斯', '逻辑回归', '随机森林', '支持向量机']:
            y_proba = results[name]['y_proba']
            fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_proba[:, idx])
            roc_auc = results[name]['metrics'][c]['auc']
            
            ax.plot(fpr, tpr, color=colors[name], lw=2, 
                   label=f'{name}', marker='o', markersize=4, alpha=0.7)
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='完美校准')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('假阳性率', fontsize=10, fontweight='bold')
        ax.set_ylabel('真阳性率', fontsize=10, fontweight='bold')
        ax.set_title(class_name, fontsize=11, fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')
    
    fig.suptitle('图4: 概率校准曲线。比较校准前后预测概率的可靠性，理想情况下应接近对角线。', 
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"各类别ROC曲线图已保存至: {save_path}")
    plt.close()


def plot_performance_comparison(results, save_path='performance_comparison.png'):
    """绘制性能对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_names = ['朴素贝叶斯', '逻辑回归', '随机森林', '支持向量机']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # 子图1: 准确率对比
    ax1 = axes[0, 0]
    accuracies = [results[name]['accuracy'] for name in model_names]
    bars = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('准确率', fontsize=12, fontweight='bold')
    ax1.set_title('各分类器准确率对比', fontsize=13, fontweight='bold')
    ax1.set_ylim([0.7, 0.9])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在条形上添加数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 子图2: F1分数对比
    ax2 = axes[0, 1]
    macro_f1s = [results[name]['macro_f1'] for name in model_names]
    weighted_f1s = [results[name]['weighted_f1'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, macro_f1s, width, label='宏平均 F1', 
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, weighted_f1s, width, label='加权平均 F1', 
                    color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('F1分数', fontsize=12, fontweight='bold')
    ax2.set_title('F1分数对比', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 子图3: AUC对比
    ax3 = axes[1, 0]
    aucs = [results[name]['macro_auc'] for name in model_names]
    bars = ax3.bar(model_names, aucs, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('宏观 AUC', fontsize=12, fontweight='bold')
    ax3.set_title('宏观AUC对比', fontsize=13, fontweight='bold')
    ax3.set_ylim([0.5, 0.9])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在条形上添加数值
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 子图4: 训练时间对比
    ax4 = axes[1, 1]
    train_times = [results[name]['train_time'] * 1000 for name in model_names]  # 转换为毫秒
    bars = ax4.bar(model_names, train_times, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('训练时间 (毫秒)', fontsize=12, fontweight='bold')
    ax4.set_title('训练时间对比', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在条形上添加数值
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    fig.suptitle('多分类器综合性能对比', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存至: {save_path}")
    plt.close()


def plot_radar_chart(results, save_path='radar_chart.png'):
    """绘制雷达图对比"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 评估指标
    categories = ['准确率', '宏平均\nF1', '加权平均\nF1', 'AUC', '训练速度']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    model_names = ['朴素贝叶斯', '逻辑回归', '随机森林', '支持向量机']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for name, color in zip(model_names, colors):
        r = results[name]
        # 归一化训练速度（越快越好，所以用1/时间）
        max_time = max(results[n]['train_time'] for n in model_names)
        speed_score = 1 - (r['train_time'] / max_time)
        
        values = [
            r['accuracy'],
            r['macro_f1'],
            r['weighted_f1'],
            r['macro_auc'],
            speed_score
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('多分类器综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存至: {save_path}")
    plt.close()


# ==================== 主程序 ====================

def main():
    print("=" * 100)
    print("实验四：高级要求 - 多分类器对比")
    print("=" * 100)
    print()
    
    # 读取数据
    print("步骤1: 读取数据...")
    X_train, y_train, feature_names = read_csv_data('train_set.csv')
    X_test, y_test, _ = read_csv_data('test_set.csv')
    
    # 转换为numpy数组（sklearn需要）
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    
    classes = sorted(list(set(y_train)))
    
    print(f"✓ 训练集: {len(X_train)} 样本")
    print(f"✓ 测试集: {len(X_test)} 样本")
    print(f"✓ 特征数量: {len(X_train[0])}")
    print(f"✓ 类别: {classes}")
    print()
    
    # 训练所有分类器
    results = train_all_classifiers(X_train_np, y_train_np, X_test_np, y_test_np, classes)
    
    # 打印对比表格
    print_comparison_table(results, classes)
    print_per_class_performance(results, classes)
    
    # 生成可视化图表
    print("\n" + "=" * 100)
    print("生成可视化图表...")
    print("=" * 100)
    print()
    
    plot_multi_model_roc_curves(results, y_test_np, classes)
    plot_per_class_roc_curves(results, y_test_np, classes)
    plot_performance_comparison(results)
    plot_radar_chart(results)
    
    # 总结
    print("\n" + "=" * 100)
    print("高级要求完成！")
    print("=" * 100)
    print()
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"最佳分类器: {best_model[0]}")
    print(f"  准确率: {best_model[1]['accuracy']:.4f}")
    print(f"  宏平均 F1: {best_model[1]['macro_f1']:.4f}")
    print(f"  加权平均 F1: {best_model[1]['weighted_f1']:.4f}")
    print(f"  宏观 AUC: {best_model[1]['macro_auc']:.4f}")
    print(f"  训练时间: {best_model[1]['train_time']:.6f}秒")
    print()
    
    print("生成的图表文件:")
    print("  1. multi_model_roc.png - 多模型ROC曲线对比（宏观平均）")
    print("  2. per_class_roc.png - 各类别ROC曲线对比（图4）")
    print("  3. performance_comparison.png - 性能指标对比")
    print("  4. radar_chart.png - 综合性能雷达图")
    print()


if __name__ == "__main__":
    main()
