# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import rotate
import random
import matplotlib.pyplot as plt

# 配置matplotlib以正确显示中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("未找到'SimHei'字体，中文可能无法正常显示。请尝试安装或更换为其他中文字体。")

def load_and_process_data(file_path):
    """
    从指定路径加载semeion数据集。
    """
    print(f"--- 正在从 '{file_path}' 加载数据 ---")
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{file_path}'。")
        return None, None
    
    features = data[:, :256]
    labels_one_hot = data[:, 256:]
    labels = np.argmax(labels_one_hot, axis=1)

    print(f"数据加载完成，总样本数: {features.shape[0]}")
    return features, labels

def augment_data_with_random_rotation(features, labels):
    """
    通过在指定范围内随机旋转来增强图像数据。
    为每个原始图像生成两个新的旋转图像。
    """
    print("--- 开始进行随机旋转数据增强 ---")
    augmented_features = []
    augmented_labels = []

    # 遍历每一张原始图像
    for img_flat, label in zip(features, labels):
        # 将一维向量重塑为 16x16 的二维图像
        img_2d = img_flat.reshape(16, 16)
        
        # 1. 生成左上方向的随机旋转 (-10° 到 -5°)
        angle1 = random.uniform(-10, -5)
        # 2. 生成左下方向的随机旋转 (5° 到 10°)
        angle2 = random.uniform(5, 10)
        
        angles_to_apply = [angle1, angle2]
        
        for angle in angles_to_apply:
            # 使用 scipy.ndimage.rotate 进行旋转
            # order=1 指定使用双线性插值
            # reshape=False 保持图像尺寸为16x16
            rotated_img = rotate(img_2d, angle, reshape=False, order=1)
            
            # 将旋转后的二维图像重新展平并存入列表
            augmented_features.append(rotated_img.flatten())
            # 对应的标签不变
            augmented_labels.append(label)

    print(f"数据增强完成，共生成 {len(augmented_features)} 个新样本。")
    return np.array(augmented_features), np.array(augmented_labels)

def save_enhanced_data(features, labels, filename):
    """
    将增强后的数据保存到文件中，格式与原始semeion.data一致。
    """
    print(f"\n--- 正在将新数据保存到 '{filename}' ---")
    
    # 1. 将单数字标签转换回 one-hot 编码
    num_classes = 10
    labels_one_hot = np.eye(num_classes, dtype=int)[labels]
    
    # 2. 将特征和 one-hot 标签水平合并
    combined_data = np.hstack((features, labels_one_hot))
    
    # 3. 定义保存格式 (256个浮点数 + 10个整数)
    fmt = ['%.8f'] * 256 + ['%d'] * 10
    
    # 4. 使用 np.savetxt 保存文件
    np.savetxt(filename, combined_data, fmt=fmt, delimiter=' ')
    
    print(f"成功！增强后的数据集已保存为 '{filename}'。")

    """
    可视化单张原始图像及其两个随机旋转后的版本。
    """
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_img.reshape(16, 16), cmap='gray')
    plt.title(f"原始图像\n标签: {original_label}")
    plt.axis('off')
    
    # 显示旋转后的图像
    plt.subplot(1, 3, 2)
    plt.imshow(augmented_imgs[0].reshape(16, 16), cmap='gray')
    plt.title("随机旋转 (-10° ~ -5°)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(augmented_imgs[1].reshape(16, 16), cmap='gray')
    plt.title("随机旋转 (5° ~ 10°)")
    plt.axis('off')
        
    plt.suptitle("随机旋转数据增强效果对比", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = "random_rotation_comparison.png"
    plt.savefig(output_filename)
    print(f"\n已保存数据增强效果对比图：{output_filename}")

    # 【核心修改】将输出文件名修改为 'new.data'
    NEW_DATA_FILE_PATH = 'new.data'

    # 1. 加载原始数据
    features, labels = load_and_process_data(DATA_FILE_PATH)
    
    if features is not None:
        # 2. 生成增强数据
        augmented_X, augmented_y = augment_data_with_random_rotation(features, labels)
        
        # 3. 打印结果摘要
        print("\n--- 数据生成结果摘要 ---")
        print(f"原始数据集大小: {features.shape[0]} 个样本")
        print(f"为每个原始样本生成了 2 个新样本")
        print(f"总计生成了 {augmented_X.shape[0]} 个新样本")
        
        # 4. 调用函数保存新的数据集
        save_enhanced_data(augmented_X, augmented_y, NEW_DATA_FILE_PATH)
        
        # 5. 随机选择一个样本进行可视化对比
        random_index = random.randint(0, len(features) - 1)
        original_sample_img = features[random_index]
        original_sample_label = labels[random_index]
        
        start_idx = random_index * 2
        corresponding_augmented_imgs = augmented_X[start_idx : start_idx + 2]
        
        visualize_comparison(original_sample_img, corresponding_augmented_imgs, original_sample_label)