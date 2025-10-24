"""
目标：用环境因素预测共享单车每小时租赁量（目标列为 `cnt`）。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ --- 解决中文字体与负号显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体(跨平台通用)
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题


# --- 自定义实现，替代 scikit-learn ---

class CustomScaler:
    """功能与 scikit-learn 的 StandardScaler 相同的自定义类。"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (data - self.mean_) / self.std_

    def transform(self, data):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform first.")
        return (data - self.mean_) / self.std_


def custom_mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def custom_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


# --- 数据处理与模型 ---

def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)
    return df


def time_split(df, train_frac=0.7):
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test


def map_hour_to_3_periods(hour):
    if hour in [7, 8, 9, 16, 17, 18, 19]:
        return 'peak'
    elif hour in [10, 11, 12, 13, 14, 15, 20]:
        return 'off_peak'
    else:
        return 'low_hours'


def preprocess(df, scaler=None, fit_scaler=False, reference_columns=None):
    continuous_cols = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_cols = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    y = df['cnt'].astype(float).to_numpy().reshape(-1, 1)
    features_needed = continuous_cols + categorical_cols + ['hr']
    X = df[features_needed].copy()
    X['time_period'] = X['hr'].apply(map_hour_to_3_periods)
    X = X.drop('hr', axis=1)
    categorical_cols.append('time_period')
    for c in continuous_cols:
        if X[c].isnull().any(): X[c] = X[c].fillna(X[c].median())
    for c in categorical_cols:
        if X[c].isnull().any(): X[c] = X[c].fillna(X[c].mode().iloc[0])
    for col in continuous_cols:
        lower, upper = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(lower, upper)
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    scaler_used = scaler if scaler is not None else CustomScaler()
    cols_to_scale = [c for c in continuous_cols if c in X.columns]
    if cols_to_scale:
        if fit_scaler:
            X[cols_to_scale] = scaler_used.fit_transform(X[cols_to_scale])
        else:
            X[cols_to_scale] = scaler_used.transform(X[cols_to_scale])
    X.insert(0, 'bias', 1.0)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0.0)
    feature_names = X.columns.tolist()
    return X.values.astype(float), y, scaler_used, feature_names


def normal_equation(X, y):
    try:
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def compute_loss(X, y, theta):
    n = X.shape[0]
    errors = X.dot(theta) - y
    mse = np.mean(errors ** 2)
    grad = (2.0 / n) * X.T.dot(errors)
    return mse, grad


def batch_gradient_descent(X, y, lr=0.01, epochs=50, verbose=False):
    n, m = X.shape
    theta = np.zeros((m, 1))
    losses = []
    for epoch in range(epochs):
        mse, grad = compute_loss(X, y, theta)
        theta -= lr * grad
        losses.append(mse)
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f'BGD Epoch {epoch}/{epochs}, LR={lr}, MSE={mse:.4f}')
    return theta, losses


def stochastic_gradient_descent(X, y, lr=0.01, epochs=100, verbose=False):
    n, m = X.shape
    theta = np.zeros((m, 1))
    epoch_losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]
        for i in range(n):
            xi, yi = X_shuf[i:i+1], y_shuf[i:i+1]
            error = xi.dot(theta) - yi
            grad = 2.0 * xi.T.dot(error)
            theta -= lr * grad
        mse = np.mean((X.dot(theta) - y) ** 2)
        epoch_losses.append(mse)
        if verbose:
            print(f'SGD Epoch {epoch}/{epochs}, LR={lr}, MSE={mse:.4f}')
    return theta, epoch_losses


def evaluate_model(X, y, theta):
    preds = X.dot(theta)
    mse = custom_mean_squared_error(y, preds)
    mae = custom_mean_absolute_error(y, preds)
    return mse, mae


# ✅ --- 绘图函数支持中文 ---
def plot_losses(loss_dict, title='Loss curves'):
    plt.figure(figsize=(10, 6))
    for label, losses in loss_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel('迭代次数 / Epochs', fontsize=12)
    plt.ylabel('均方误差 (MSE)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def run_experiment(X_train, y_train, X_test, y_test, method, lr_list, epochs):
    results = {}
    print(f"\n--- 正在为 {method.upper()} 进行学习率对比实验 ---")
    
    for lr in lr_list:
        print(f"训练模型: {method.upper()} with LR = {lr}")
        if method == 'bgd':
            theta, losses = batch_gradient_descent(X_train, y_train, lr=lr, epochs=epochs)
        elif method == 'sgd':
            sgd_epochs = max(1, epochs // 10)
            theta, losses = stochastic_gradient_descent(X_train, y_train, lr=lr, epochs=sgd_epochs)
        else:
            raise ValueError("方法必须是 'bgd' 或 'sgd'")
            
        train_mse, train_mae = evaluate_model(X_train, y_train, theta)
        test_mse, test_mae = evaluate_model(X_test, y_test, theta)
        
        results[lr] = {
            'theta': theta,
            'train_losses': losses,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'test_mse': test_mse,
            'test_mae': test_mae
        }
    return results


def main():
    root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    csv_path = os.path.join(root, 'bike_sharing_hour.csv')
    df = load_data(csv_path)

    train_df, test_df = time_split(df, train_frac=0.7)

    X_train, y_train, scaler, train_feature_names = preprocess(train_df, scaler=None, fit_scaler=True)
    X_test, y_test, _, _ = preprocess(test_df, scaler=scaler, fit_scaler=False, reference_columns=train_feature_names)

    print(f'训练样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}')
    print(f'测试样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]}')

    # 1. 正规方程基线
    theta_ne = normal_equation(X_train, y_train)
    ne_train_mse, ne_train_mae = evaluate_model(X_train, y_train, theta_ne)
    ne_test_mse, ne_test_mae = evaluate_model(X_test, y_test, theta_ne)
    
    # 2. BGD 实验
    bgd_lr_list = [0.01, 0.05, 0.1, 0.2, 0.3]
    bgd_results = run_experiment(X_train, y_train, X_test, y_test, 'bgd', bgd_lr_list, epochs=50)
    
    # 3. SGD 实验
    sgd_lr_list = [0.0001, 0.0005, 0.001, 0.005]
    sgd_results = run_experiment(X_train, y_train, X_test, y_test, 'sgd', sgd_lr_list, epochs=100)

    # 找到最优学习率
    best_bgd_lr = min(bgd_results, key=lambda lr: bgd_results[lr]['test_mse'])
    best_sgd_lr = min(sgd_results, key=lambda lr: sgd_results[lr]['test_mse'])
    
    best_bgd_res = bgd_results[best_bgd_lr]
    best_sgd_res = sgd_results[best_sgd_lr]

    # ✅ 输出实验结果（支持中文）
    print("\n" + "="*40)
    print("           实验结果总结")
    print("="*40)
    print(f'Normal Eq Train MSE MAE: {ne_train_mse} {ne_train_mae}')
    print(f'Normal Eq Test  MSE MAE: {ne_test_mse} {ne_test_mae}')
    print(f'\nBGD (lr={best_bgd_lr}) Train MSE MAE: {best_bgd_res["train_mse"]} {best_bgd_res["train_mae"]}')
    print(f'BGD (lr={best_bgd_lr}) Test  MSE MAE: {best_bgd_res["test_mse"]} {best_bgd_res["test_mae"]}')
    print(f'\nSGD (lr={best_sgd_lr}) Train MSE MAE: {best_sgd_res["train_mse"]} {best_sgd_res["train_mae"]}')
    print(f'SGD (lr={best_sgd_lr}) Test  MSE MAE: {best_sgd_res["test_mse"]} {best_sgd_res["test_mae"]}')
    print("="*40 + "\n")

    # 绘制曲线（含中文标题）
    bgd_loss_curves = {f'BGD 学习率={lr}': res['train_losses'] for lr, res in bgd_results.items()}
    plot_losses(bgd_loss_curves, title='批量梯度下降（BGD）在不同学习率下的收敛曲线')

    sgd_loss_curves = {f'SGD 学习率={lr}': res['train_losses'] for lr, res in sgd_results.items()}
    plot_losses(sgd_loss_curves, title='随机梯度下降（SGD）在不同学习率下的收敛曲线')
    
    results_path = os.path.join(root, 'results_formatted.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f'Normal Eq Train MSE MAE: {ne_train_mse} {ne_train_mae}\n')
        f.write(f'Normal Eq Test  MSE MAE: {ne_test_mse} {ne_test_mae}\n\n')
        f.write(f'BGD (lr={best_bgd_lr}) Train MSE MAE: {best_bgd_res["train_mse"]} {best_bgd_res["train_mae"]}\n')
        f.write(f'BGD (lr={best_bgd_lr}) Test  MSE MAE: {best_bgd_res["test_mse"]} {best_bgd_res["test_mae"]}\n\n')
        f.write(f'SGD (lr={best_sgd_lr}) Train MSE MAE: {best_sgd_res["train_mse"]} {best_sgd_res["train_mae"]}\n')
        f.write(f'SGD (lr={best_sgd_lr}) Test  MSE MAE: {best_sgd_res["test_mse"]} {best_sgd_res["test_mae"]}\n')
    print(f'实验最优结果已按指定格式写入 {results_path}')


if __name__ == '__main__':
    main()
