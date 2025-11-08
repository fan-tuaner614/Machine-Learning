import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
# ------------------------------------

# ==============================
# 1. 汇总所有实验的最佳结果
# ==============================

methods_A = [
    "Parameter (LRT)",
    "Parameter (MAP)",
    "Kernel (LRT, h=2.0)", 
    "Kernel (MAP, h=2.0)", 
    "K-NN (LRT, k=3)",     
    "K-NN (MAP, k=3)"      
]
accuracies_A = np.array([
    0.8367, 
    0.8367, 
    0.8383,
    0.8383, 
    0.8667, 
    0.8667 
])

methods_B = [
    "Parameter (LRT)",
    "Parameter (MAP)",
    "Kernel (LRT, h=0.8)", 
    "Kernel (MAP, h=0.5)", 
    "K-NN (LRT, k=3)",     
    "K-NN (MAP, k=3)"      
]
accuracies_B = np.array([
    0.8725,
    0.8808, 
    0.8808, 
    0.8883, 
    0.8967, 
    0.9050  
])

methods_C = [
    "Parameter (LRT)",
    "Parameter (MAP)",
    "Kernel (LRT, h=1.5)", 
    "Kernel (MAP, h=0.8)", 
    "K-NN (LRT, k=3)",     
    "K-NN (MAP, k=3)"      
]
accuracies_C = np.array([
    0.8400, 
    0.8783, 
    0.8483, 
    0.8817, 
    0.8667, 
    0.9000  
])

dataset_info = [
    ("Dataset A (均衡先验)", methods_A, accuracies_A),
    ("Dataset B (偏斜先验1)", methods_B, accuracies_B),
    ("Dataset C (偏斜先验2)", methods_C, accuracies_C)
]

# ==============================
# 2. 绘制并排的三张图
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True) 
fig.suptitle("三种估计方法在三个数据集上的最终性能对比", fontsize=20, weight='bold')

all_accs = np.concatenate([accuracies_A, accuracies_B, accuracies_C])
y_min = all_accs.min() * 0.99 
y_max = all_accs.max() * 1.01 

for ax, (name, methods, accs) in zip(axes, dataset_info):
    bars = ax.bar(methods, accs, color='mediumturquoise', edgecolor='k', alpha=0.7)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{acc:.4f}", ha='center', va='bottom', fontsize=10) 

    ax.set_title(f"{name}", fontsize=16)
    if ax == axes[0]:
        ax.set_ylabel("Accuracy", fontsize=12)
        
    ax.set_ylim(y_min, y_max)  
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=55, ha='right', fontsize=10) 

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()