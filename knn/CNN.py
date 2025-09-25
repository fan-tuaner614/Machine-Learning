# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    print(f"--- 正在从 '{file_path}' 加载数据 ---")
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{file_path}'。")
        return None, None
    features_flat = data[:, :256]
    labels_one_hot = data[:, 256:]
    labels = np.argmax(labels_one_hot, axis=1)
    features_cnn = features_flat.reshape(-1, 16, 16, 1)
    print("数据加载和预处理完成。")
    return features_cnn, labels

def build_cnn_model(input_shape=(16, 16, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def calculate_cen(y_true, y_pred, num_classes=10):
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    total_cen = 0.0
    total_samples = len(y_true)
    for i in range(num_classes):
        row = conf_matrix[i, :]
        row_sum = np.sum(row)
        if row_sum == 0: continue
        class_entropy = 0.0
        for count in row:
            if count > 0:
                prob = count / row_sum
                class_entropy -= prob * np.log2(prob)
        total_cen += (row_sum / total_samples) * class_entropy
    return total_cen

def plot_confusion_matrix(y_true, y_pred, title):
    print("\n--- 正在生成混淆矩阵可视化图 ---")
    num_classes = len(np.unique(y_true))
    class_names = [str(i) for i in range(num_classes)]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('真实标签 (True Label)', fontsize=12)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
    plt.show()

if __name__ == "__main__":
    DATA_FILE_PATH = 'new.data'
    X, y = load_and_preprocess_data(DATA_FILE_PATH)

    if X is not None:
        N_SPLITS = 5
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        all_true_labels, all_predictions = [], []
        fold_metrics = {"accuracy": [], "nmi": [], "cen": []}
        start_time = time.time()
        
        print(f"\n--- 开始 {N_SPLITS}-折交叉验证 ---")
        
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"\n--- 第 {fold_idx + 1}/{N_SPLITS} 折 ---")
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_onehot = to_categorical(y_train, num_classes=10)

            print(f"正在训练模型 (训练集大小: {len(X_train)}, 测试集大小: {len(X_test)})...")
            model = build_cnn_model() 
            model.fit(X_train, y_train_onehot, epochs=15, batch_size=32, verbose=0)

            pred_probs = model.predict(X_test, verbose=0)
            predictions = np.argmax(pred_probs, axis=1)
            
            all_true_labels.extend(y_test)
            all_predictions.extend(predictions)

            acc = metrics.accuracy_score(y_test, predictions)
            nmi = metrics.normalized_mutual_info_score(y_test, predictions)
            cen = calculate_cen(y_test, predictions)
            
            fold_metrics["accuracy"].append(acc)
            fold_metrics["nmi"].append(nmi)
            fold_metrics["cen"].append(cen)
            
            print(f"第 {fold_idx + 1} 折评估完成: Accuracy={acc:.2%}, NMI={nmi:.4f}, CEN={cen:.4f}")

        end_time = time.time()
        print(f"\n--- {N_SPLITS}-折交叉验证完成 ---")

        print("\n================== 模型最终K-Fold评估结果 ==================")
        print(f"总运行时间: {(end_time - start_time) / 60:.2f} 分钟")
        print(f"平均准确率 (Accuracy): {np.mean(fold_metrics['accuracy']):.2%} ± {np.std(fold_metrics['accuracy']):.2%}")
        print(f"平均归一化互信息 (NMI): {np.mean(fold_metrics['nmi']):.4f} ± {np.std(fold_metrics['nmi']):.4f}")
        print(f"平均混淆熵 (CEN): {np.mean(fold_metrics['cen']):.4f} ± {np.std(fold_metrics['cen']):.4f}")
        print("=========================================================")

        plot_confusion_matrix(all_true_labels, all_predictions, f'{N_SPLITS}-折交叉验证的混淆矩阵 (汇总)')