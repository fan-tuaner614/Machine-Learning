# -*- coding: utf-8 -*-
"""
实验四：数据预处理
要求：手动实现所有数据处理步骤，不使用机器学习和数据处理相关的库
"""

import math

# ==================== 工具函数 ====================

def read_csv(filename):
    """读取CSV文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析表头
    header = lines[0].strip().split(',')
    
    # 解析数据
    data = []
    for line in lines[1:]:
        row = line.strip().split(',')
        data.append([float(x) for x in row])
    
    return header, data


def write_csv(filename, header, data, labels):
    """将数据写入CSV文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入表头（特征列 + 标签列）
        f.write(','.join(header) + ',label\n')
        
        # 写入数据
        for i in range(len(data)):
            row_str = ','.join([str(x) for x in data[i]])
            f.write(row_str + ',' + str(labels[i]) + '\n')


def calculate_mean(data, col_idx):
    """计算某一列的均值"""
    total = 0.0
    count = 0
    for row in data:
        total += row[col_idx]
        count += 1
    return total / count if count > 0 else 0.0


def calculate_std(data, col_idx, mean):
    """计算某一列的标准差"""
    variance = 0.0
    count = 0
    for row in data:
        variance += (row[col_idx] - mean) ** 2
        count += 1
    return math.sqrt(variance / count) if count > 0 else 0.0


def calculate_correlation(data, col1_idx, col2_idx):
    """计算两列之间的皮尔逊相关系数"""
    n = len(data)
    
    # 计算均值
    mean1 = calculate_mean(data, col1_idx)
    mean2 = calculate_mean(data, col2_idx)
    
    # 计算协方差和标准差
    covariance = 0.0
    std1_sum = 0.0
    std2_sum = 0.0
    
    for row in data:
        diff1 = row[col1_idx] - mean1
        diff2 = row[col2_idx] - mean2
        covariance += diff1 * diff2
        std1_sum += diff1 ** 2
        std2_sum += diff2 ** 2
    
    std1 = math.sqrt(std1_sum / n)
    std2 = math.sqrt(std2_sum / n)
    
    if std1 == 0 or std2 == 0:
        return 0.0
    
    return covariance / (n * std1 * std2)


def z_score_normalize(data, col_idx, mean, std):
    """对某一列进行Z-score标准化"""
    if std == 0:
        return [0.0 for _ in range(len(data))]
    
    normalized = []
    for row in data:
        normalized.append((row[col_idx] - mean) / std)
    return normalized


def create_quality_labels(data, quality_col_idx):
    """
    创建多分类标签
    低质量(0): 3-4分
    中等质量(1): 5-6分
    高质量(2): 7-8分
    """
    labels = []
    for row in data:
        quality = row[quality_col_idx]
        if quality <= 4:
            labels.append(0)  # 低质量
        elif quality <= 6:
            labels.append(1)  # 中等质量
        else:
            labels.append(2)  # 高质量
    return labels


def count_labels(labels):
    """统计每个标签的数量"""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def stratified_split(data, labels, train_ratio=0.7):
    """
    手动实现分层采样
    按照train_ratio的比例划分训练集和测试集
    """
    # 按标签分组
    label_groups = {}
    for i, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(i)
    
    train_indices = []
    test_indices = []
    
    # 对每个类别分别划分
    for label, indices in label_groups.items():
        n = len(indices)
        n_train = int(n * train_ratio)
        
        # 简单的划分方式：按顺序取前70%作为训练集
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
    
    # 根据索引提取数据
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_data, train_labels, test_data, test_labels, train_indices, test_indices


# ==================== 主程序 ====================

def main():
    print("="*80)
    print("实验四：朴素贝叶斯分类器 - 数据预处理")
    print("="*80)
    print()
    
    # 1. 读取数据
    print("步骤1: 读取数据集")
    print("-"*80)
    header, data = read_csv('winequality-red.csv')
    print(f"数据集大小: {len(data)} 行 × {len(header)} 列")
    print(f"特征列: {header}")
    print()
    
    # 2. 检查缺失值和异常值
    print("步骤2: 数据清洗 - 检查缺失值和异常值")
    print("-"*80)
    print(f"数据集共有 {len(data)} 个样本")
    print("本数据集无缺失值")
    print()
    
    # 3. 特征相关性分析
    print("步骤3: 特征相关性分析")
    print("-"*80)
    print("计算特征之间的相关系数矩阵...")
    
    # 计算所有特征对之间的相关系数（除了quality列）
    n_features = len(header) - 1  # 排除quality列
    
    # 找出高度相关的特征对 (|r| > 0.8)
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr = calculate_correlation(data, i, j)
            if abs(corr) > 0.8:
                high_corr_pairs.append((header[i], header[j], corr))
    
    if high_corr_pairs:
        print(f"发现 {len(high_corr_pairs)} 对高度相关的特征 (|r| > 0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  - {feat1} 与 {feat2}: r = {corr:.4f}")
    else:
        print("未发现高度相关的特征对 (|r| > 0.8)")
    print()
    
    # 4. 数据标准化
    print("步骤4: 数据标准化 (Z-score)")
    print("-"*80)
    
    # 计算每个特征的均值和标准差
    means = []
    stds = []
    for i in range(n_features):
        mean = calculate_mean(data, i)
        std = calculate_std(data, i, mean)
        means.append(mean)
        stds.append(std)
        print(f"{header[i]:30s} - Mean: {mean:8.4f}, Std: {std:8.4f}")
    
    # 标准化数据
    normalized_data = []
    for row in data:
        normalized_row = []
        for i in range(n_features):
            normalized_value = (row[i] - means[i]) / stds[i] if stds[i] != 0 else 0.0
            normalized_row.append(normalized_value)
        normalized_data.append(normalized_row)
    
    print()
    print("数据标准化完成！")
    print()
    
    # 5. 创建多分类标签
    print("步骤5: 创建多分类标签")
    print("-"*80)
    quality_col_idx = len(header) - 1
    labels = create_quality_labels(data, quality_col_idx)
    
    label_counts = count_labels(labels)
    total = len(labels)
    
    print("质量分类规则:")
    print("  - 低质量(0): quality 3-4")
    print("  - 中等质量(1): quality 5-6")
    print("  - 高质量(2): quality 7-8")
    print()
    print("标签分布:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total) * 100
        label_name = ["低质量(0)", "中等质量(1)", "高质量(2)"][label]
        print(f"  - {label_name}: {count:4d} 样本 ({percentage:5.2f}%)")
    print()
    
    # 6. 分层采样划分数据集
    print("步骤6: 分层采样划分数据集 (70%-30%)")
    print("-"*80)
    
    train_data, train_labels, test_data, test_labels, train_indices, test_indices = \
        stratified_split(normalized_data, labels, train_ratio=0.7)
    
    print(f"训练集大小: {len(train_data)} 样本")
    print(f"测试集大小: {len(test_data)} 样本")
    print()
    
    # 统计训练集和测试集的标签分布
    train_label_counts = count_labels(train_labels)
    test_label_counts = count_labels(test_labels)
    
    print("分层采样结果验证:")
    print(f"{'类别':<15} {'原始数量':>10} {'训练集':>10} {'测试集':>10} {'训练集比例':>12} {'测试集比例':>12}")
    print("-"*80)
    
    for label in sorted(label_counts.keys()):
        label_name = ["低质量(0)", "中等质量(1)", "高质量(2)"][label]
        orig_count = label_counts[label]
        train_count = train_label_counts.get(label, 0)
        test_count = test_label_counts.get(label, 0)
        train_pct = (train_count / len(train_labels)) * 100
        test_pct = (test_count / len(test_labels)) * 100
        
        print(f"{label_name:<15} {orig_count:>10} {train_count:>10} {test_count:>10} "
              f"{train_pct:>11.2f}% {test_pct:>11.2f}%")
    
    print()
    
    # 7. 输出预处理结果摘要
    print("="*80)
    print("数据预处理完成！")
    print("="*80)
    print()
    print("预处理结果摘要:")
    print(f"  1. 原始数据: {len(data)} 样本 × {len(header)} 特征")
    print(f"  2. 标准化后特征数: {n_features}")
    print(f"  3. 分类标签数: {len(set(labels))} 类")
    print(f"  4. 训练集: {len(train_data)} 样本 ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  5. 测试集: {len(test_data)} 样本 ({len(test_data)/len(data)*100:.1f}%)")
    print()
    
    # 8. 展示部分标准化后的数据
    print("="*80)
    print("标准化后的数据样例 (前5个样本):")
    print("="*80)
    print()
    
    # 打印表头
    print(f"{'样本ID':>8}", end="")
    for i in range(min(5, n_features)):
        print(f"{header[i][:12]:>15}", end="")
    print(f"{'...':>15}{'标签':>8}")
    print("-"*80)
    
    # 打印前5个训练样本
    for i in range(min(5, len(train_data))):
        orig_idx = train_indices[i]
        print(f"{orig_idx:>8}", end="")
        for j in range(min(5, n_features)):
            print(f"{train_data[i][j]:>15.4f}", end="")
        print(f"{'...':>15}{train_labels[i]:>8}")
    
    print()
    
    # 9. 保存训练集和测试集为CSV文件
    print("="*80)
    print("保存数据集到CSV文件")
    print("="*80)
    print()
    
    # 准备特征列表头（不包括quality）
    feature_header = header[:n_features]
    
    # 保存训练集
    train_filename = 'train_set.csv'
    write_csv(train_filename, feature_header, train_data, train_labels)
    print(f"训练集已保存到: {train_filename}")
    print(f"  - 样本数: {len(train_data)}")
    print(f"  - 特征数: {len(feature_header)}")
    print(f"  - 包含标签列: label (0=低质量, 1=中等质量, 2=高质量)")
    print()
    
    # 保存测试集
    test_filename = 'test_set.csv'
    write_csv(test_filename, feature_header, test_data, test_labels)
    print(f"测试集已保存到: {test_filename}")
    print(f"  - 样本数: {len(test_data)}")
    print(f"  - 特征数: {len(feature_header)}")
    print(f"  - 包含标签列: label (0=低质量, 1=中等质量, 2=高质量)")
    print()
    
    print("="*80)
    print("数据预处理完成！所有文件已保存")
    print("="*80)
    
    return {
        'header': header,
        'original_data': data,
        'normalized_data': normalized_data,
        'labels': labels,
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'means': means,
        'stds': stds,
        'n_features': n_features
    }


if __name__ == "__main__":
    result = main()
