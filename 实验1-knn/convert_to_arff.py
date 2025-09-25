# -*- coding: utf-8 -*-
import numpy as np
import os

def convert_data_to_arff(input_file, output_file):
    """
    将 semeion.data 格式的文件转换为 Weka ARFF 格式。

    Args:
        input_file (str): 输入的 .data 文件路径。
        output_file (str): 输出的 .arff 文件路径。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在。")
        return

    print(f"开始转换 '{input_file}' -> '{output_file}'...")

    # --- 1. 定义 ARFF 文件头 ---
    # ARFF 文件头包含关系名、属性定义和数据声明
    arff_header = []
    arff_header.append("@RELATION SemeionHandwrittenDigit")
    arff_header.append("") # 空行以增加可读性

    # 添加256个像素属性 (pixel_1 到 pixel_256)
    for i in range(1, 257):
        arff_header.append(f"@ATTRIBUTE pixel_{i} NUMERIC")
    
    # 添加一个名为 'class' 的类别属性，它可以是0到9中的任意一个值
    class_labels = ",".join([str(i) for i in range(10)])
    arff_header.append(f"@ATTRIBUTE class {{{class_labels}}}")
    arff_header.append("")
    arff_header.append("@DATA")
    
    # --- 2. 加载并处理数据 ---
    try:
        data = np.loadtxt(input_file)
        features = data[:, :256]  # 前256列是像素
        labels_one_hot = data[:, 256:] # 后10列是one-hot标签
        
        # 将 one-hot 标签转换为单个数字
        # 例如 [0,0,1,0,...] -> 2
        labels = np.argmax(labels_one_hot, axis=1)

    except Exception as e:
        print(f"读取或处理文件时出错: {e}")
        return

    # --- 3. 写入 ARFF 文件 ---
    try:
        with open(output_file, 'w') as f:
            # 首先写入文件头
            for line in arff_header:
                f.write(line + "\n")
            
            # 接着逐行写入数据
            for i in range(len(features)):
                # 将256个像素值转换为逗号分隔的字符串
                feature_str = ",".join(map(str, features[i]))
                # 获取对应的标签
                label_str = str(labels[i])
                
                # 组合成最终的 ARFF 数据行
                # 格式: pixel_1,pixel_2,...,pixel_256,class_label
                data_line = f"{feature_str},{label_str}"
                f.write(data_line + "\n")

        print(f"转换成功！已生成文件 '{output_file}'。")

    except Exception as e:
        print(f"写入 ARFF 文件时出错: {e}")

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    input_data_file = 'semeion.data'
    output_arff_file = 'semeion.arff'
    
    convert_data_to_arff(input_data_file, output_arff_file)