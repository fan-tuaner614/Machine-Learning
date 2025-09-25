import numpy as np

def calculate_cen_from_matrix(conf_matrix):
    """
    根据给定的混淆矩阵计算混淆熵 (CEN)。
    
    参数:
    conf_matrix (np.array): 一个N x N的numpy数组形式的混淆矩阵。
    
    返回:
    float: 计算出的混淆熵值。
    """
    # 1. 计算总样本数
    total_samples = np.sum(conf_matrix)
    if total_samples == 0:
        return 0

    total_cen = 0.0
    
    # 2. 遍历矩阵的每一行（每个真实类别）
    for row in conf_matrix:
        row_sum = np.sum(row)
        if row_sum == 0:
            continue  # 如果某个类别没有样本，则跳过

        class_entropy = 0.0
        # 3. 计算该行的信息熵
        for count in row:
            if count > 0:
                prob = count / row_sum
                class_entropy -= prob * np.log2(prob)
        
        # 4. 将该行的熵进行加权，并累加到总CEN中
        weight = row_sum / total_samples
        total_cen += weight * class_entropy
        
    return total_cen

# --- 主程序 ---
# 将您图片中的混淆矩阵数据转录为numpy数组
# 行对应真实类别 a=0, b=1, ..., j=9
# 列对应预测类别 a=0, b=1, ..., j=9
confusion_matrix_data = np.array([
    [158, 0, 0, 0, 1, 0, 2, 0, 0, 0],    # a = 0
    [0, 158, 1, 2, 0, 0, 0, 1, 0, 0],    # b = 1
    [0, 5, 151, 1, 1, 0, 0, 0, 1, 0],    # c = 2
    [0, 4, 0, 150, 0, 3, 0, 1, 1, 0],    # d = 3
    [0, 12, 0, 0, 144, 0, 1, 3, 0, 1],   # e = 4
    [3, 0, 0, 1, 1, 149, 5, 0, 0, 0],    # f = 5
    [6, 1, 0, 0, 1, 2, 152, 0, 0, 0],    # g = 6
    [0, 21, 0, 0, 1, 0, 1, 134, 0, 1],   # h = 7
    [1, 3, 11, 7, 0, 8, 2, 0, 121, 2],   # i = 8
    [4, 9, 0, 19, 1, 9, 1, 0, 4, 111]    # j = 9
])


# 调用函数进行计算
cen_value = calculate_cen_from_matrix(confusion_matrix_data)

# 打印结果
print(f"根据提供的混淆矩阵，计算出的混淆熵 (CEN) 为: {cen_value:.4f}")