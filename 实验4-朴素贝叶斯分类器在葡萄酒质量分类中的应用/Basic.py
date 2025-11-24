# -*- coding: utf-8 -*-
"""
实验四：朴素贝叶斯分类器实现与性能评估
要求：手动实现朴素贝叶斯分类器，不使用机器学习库
"""

import math
import time


# ==================== 数据读取函数 ====================

def read_csv_data(filename):
    """读取CSV格式的训练集或测试集"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析表头
    header = lines[0].strip().split(',')
    
    # 解析数据和标签
    data = []
    labels = []
    for line in lines[1:]:
        row = line.strip().split(',')
        if row:
            data.append([float(x) for x in row[:-1]])  # 特征数据
            labels.append(int(row[-1]))  # 标签
    
    return data, labels


# ==================== 高斯朴素贝叶斯分类器 ====================

class GaussianNaiveBayes:
    """手动实现的高斯朴素贝叶斯分类器"""
    
    def __init__(self):
        self.classes = []  # 类别列表
        self.class_priors = {}  # 类先验概率 P(c)
        self.feature_stats = {}  # 特征统计信息 {class: {feature_idx: (mean, std)}}
        
    def fit(self, X_train, y_train):
        """
        训练高斯朴素贝叶斯分类器
        
        参数:
            X_train: 训练数据，列表形式 [[特征1, 特征2, ...], ...]
            y_train: 训练标签，列表形式 [标签1, 标签2, ...]
        """
        n_samples = len(X_train)
        n_features = len(X_train[0])
        
        # 统计每个类别的样本
        class_samples = {}
        for i in range(n_samples):
            label = y_train[i]
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(X_train[i])
        
        self.classes = sorted(class_samples.keys())
        
        # 计算类先验概率 P(c)
        for c in self.classes:
            self.class_priors[c] = len(class_samples[c]) / n_samples
        
        # 计算每个类别、每个特征的均值和标准差
        for c in self.classes:
            self.feature_stats[c] = {}
            samples = class_samples[c]
            
            for feature_idx in range(n_features):
                # 提取该特征的所有值
                feature_values = [sample[feature_idx] for sample in samples]
                
                # 计算均值
                mean = sum(feature_values) / len(feature_values)
                
                # 计算标准差
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                std = math.sqrt(variance) if variance > 0 else 1e-6  # 避免除零
                
                self.feature_stats[c][feature_idx] = (mean, std)
    
    def _gaussian_probability(self, x, mean, std):
        """
        计算高斯概率密度
        P(x) = (1 / sqrt(2π * σ²)) * exp(-(x - μ)² / (2σ²))
        """
        if std == 0:
            std = 1e-6
        
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    
    def _calculate_log_probability(self, sample, c):
        """
        计算样本属于类别c的对数概率
        使用对数避免数值下溢
        
        log P(c|x) ∝ log P(c) + Σ log P(xi|c)
        """
        # 类先验概率的对数
        log_prob = math.log(self.class_priors[c])
        
        # 累加各特征的对数条件概率
        for feature_idx in range(len(sample)):
            mean, std = self.feature_stats[c][feature_idx]
            prob = self._gaussian_probability(sample[feature_idx], mean, std)
            
            # 避免log(0)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += -1e10  # 使用一个很小的负数
        
        return log_prob
    
    def predict(self, X_test):
        """
        预测测试数据的类别
        
        参数:
            X_test: 测试数据
        
        返回:
            预测标签列表
        """
        predictions = []
        
        for sample in X_test:
            # 计算该样本属于每个类别的概率
            class_probs = {}
            for c in self.classes:
                class_probs[c] = self._calculate_log_probability(sample, c)
            
            # 选择概率最大的类别
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        预测测试数据属于每个类别的概率
        
        返回:
            概率列表 [[P(c0), P(c1), P(c2)], ...]
        """
        probabilities = []
        
        for sample in X_test:
            # 计算对数概率
            log_probs = {}
            for c in self.classes:
                log_probs[c] = self._calculate_log_probability(sample, c)
            
            # 转换为概率（使用softmax避免数值问题）
            max_log_prob = max(log_probs.values())
            exp_probs = {c: math.exp(log_probs[c] - max_log_prob) for c in self.classes}
            total = sum(exp_probs.values())
            
            probs = [exp_probs[c] / total for c in self.classes]
            probabilities.append(probs)
        
        return probabilities


# ==================== 性能评估函数 ====================

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, classes):
    """
    计算混淆矩阵
    
    返回:
        二维列表，matrix[i][j]表示真实类别i被预测为类别j的数量
    """
    n_classes = len(classes)
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    
    for i in range(len(y_true)):
        true_idx = classes.index(y_true[i])
        pred_idx = classes.index(y_pred[i])
        matrix[true_idx][pred_idx] += 1
    
    return matrix


def print_confusion_matrix(matrix, classes):
    """打印混淆矩阵"""
    print("\n混淆矩阵:")
    print("-" * 60)
    
    # 打印表头
    print(f"{'真实\\预测':<15}", end="")
    for c in classes:
        print(f"{c:>10}", end="")
    print()
    print("-" * 60)
    
    # 打印每一行
    class_names = ["低质量(0)", "中等质量(1)", "高质量(2)"]
    for i, c in enumerate(classes):
        print(f"{class_names[i]:<15}", end="")
        for j in range(len(classes)):
            print(f"{matrix[i][j]:>10}", end="")
        print()
    print()


# ==================== scikit-learn对比 ====================

def compare_with_sklearn(X_train, y_train, X_test, y_test):
    """与scikit-learn的实现进行对比"""
    try:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        
        # 使用sklearn训练
        start_time = time.time()
        sklearn_nb = GaussianNB()
        sklearn_nb.fit(X_train, y_train)
        sklearn_pred = sklearn_nb.predict(X_test)
        sklearn_time = time.time() - start_time
        
        sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
        
        return sklearn_accuracy, sklearn_time, True
    except ImportError:
        print("警告: 未安装scikit-learn，无法进行对比")
        return None, None, False


# ==================== 主程序 ====================

def main():
    print("=" * 80)
    print("实验四：朴素贝叶斯分类器实现与基础性能评估")
    print("=" * 80)
    print()
    
    # 1. 读取数据
    print("步骤1: 读取训练集和测试集")
    print("-" * 80)
    X_train, y_train = read_csv_data('train_set.csv')
    X_test, y_test = read_csv_data('test_set.csv')
    
    print(f"训练集: {len(X_train)} 样本, {len(X_train[0])} 特征")
    print(f"测试集: {len(X_test)} 样本, {len(X_test[0])} 特征")
    print()
    
    # 2. 训练手动实现的朴素贝叶斯分类器
    print("步骤2: 训练手动实现的高斯朴素贝叶斯分类器")
    print("-" * 80)
    
    start_time = time.time()
    custom_nb = GaussianNaiveBayes()
    custom_nb.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"训练完成！用时: {train_time:.4f} 秒")
    print()
    
    # 打印学到的参数
    print("类先验概率:")
    class_names = ["低质量(0)", "中等质量(1)", "高质量(2)"]
    for i, c in enumerate(custom_nb.classes):
        print(f"  {class_names[i]}: P(c={c}) = {custom_nb.class_priors[c]:.4f}")
    print()
    
    # 3. 预测
    print("步骤3: 在测试集上进行预测")
    print("-" * 80)
    
    start_time = time.time()
    custom_pred = custom_nb.predict(X_test)
    predict_time = time.time() - start_time
    
    print(f"预测完成！用时: {predict_time:.4f} 秒")
    print()
    
    # 4. 计算准确率
    print("步骤4: 计算分类准确率")
    print("-" * 80)
    
    custom_accuracy = calculate_accuracy(y_test, custom_pred)
    print(f"手动实现的朴素贝叶斯分类器准确率: {custom_accuracy:.4f} ({custom_accuracy*100:.2f}%)")
    print()
    
    # 5. 混淆矩阵
    print("步骤5: 生成混淆矩阵")
    print("-" * 80)
    
    cm = confusion_matrix(y_test, custom_pred, custom_nb.classes)
    print_confusion_matrix(cm, custom_nb.classes)
    
    # 6. 与scikit-learn对比
    print("步骤6: 与scikit-learn库的性能对比")
    print("-" * 80)
    
    sklearn_accuracy, sklearn_time, sklearn_available = compare_with_sklearn(
        X_train, y_train, X_test, y_test
    )
    
    # 7. 性能对比表格
    print("\n" + "=" * 80)
    print("性能对比表格")
    print("=" * 80)
    print()
    
    print(f"{'实现方式':<25} {'训练时间(秒)':>15} {'预测时间(秒)':>15} {'准确率':>15} {'准确率(%)':>15}")
    print("-" * 80)
    
    print(f"{'手动实现':<25} {train_time:>15.6f} {predict_time:>15.6f} "
          f"{custom_accuracy:>15.4f} {custom_accuracy*100:>15.2f}")
    
    if sklearn_available:
        print(f"{'scikit-learn':<25} {sklearn_time:>15.6f} {'N/A':>15} "
              f"{sklearn_accuracy:>15.4f} {sklearn_accuracy*100:>15.2f}")
        
        print()
        print("准确率差异:")
        diff = abs(custom_accuracy - sklearn_accuracy)
        print(f"  绝对差异: {diff:.6f}")
        print(f"  相对差异: {(diff/sklearn_accuracy)*100:.4f}%")
        
        if diff < 0.001:
            print("  结论: 手动实现与scikit-learn实现结果基本一致 ✓")
        else:
            print(f"  结论: 存在 {diff*100:.2f}% 的差异")
    else:
        print(f"{'scikit-learn':<25} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
        print("\n注: scikit-learn未安装，无法进行对比")
    
    print()
    
    # 8. 详细分析
    print("=" * 80)
    print("分类结果详细分析")
    print("=" * 80)
    print()
    
    # 统计每个类别的预测情况
    for i, c in enumerate(custom_nb.classes):
        true_count = sum(1 for label in y_test if label == c)
        pred_count = sum(1 for label in custom_pred if label == c)
        correct_count = sum(1 for j in range(len(y_test)) if y_test[j] == c and custom_pred[j] == c)
        
        print(f"{class_names[i]}:")
        print(f"  真实样本数: {true_count}")
        print(f"  预测样本数: {pred_count}")
        print(f"  正确预测数: {correct_count}")
        if true_count > 0:
            print(f"  召回率: {correct_count/true_count:.4f} ({correct_count/true_count*100:.2f}%)")
        if pred_count > 0:
            print(f"  精确率: {correct_count/pred_count:.4f} ({correct_count/pred_count*100:.2f}%)")
        print()
    
    print("=" * 80)
    print("基础性能评估完成！")
    print("=" * 80)
    
    return {
        'custom_nb': custom_nb,
        'custom_accuracy': custom_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    result = main()
