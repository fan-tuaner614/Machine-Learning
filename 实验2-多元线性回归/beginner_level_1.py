"""
目标：用环境因素预测共享单车每小时租赁量（目标列为 `cnt`）。

改动要点：
- 继续移除 weekday 特征。
- 分别绘制 BGD 与 SGD 的收敛曲线（两张图）。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- 自定义实现（替代 sklearn）---

class CustomScaler:
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
            raise RuntimeError("Scaler not fitted.")
        return (data - self.mean_) / self.std_


def custom_mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def custom_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


# --- 数据加载与预处理 ---

def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)
    return df


def time_split(df, train_frac=0.7):
    n = len(df)
    split = int(n * train_frac)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


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

    # 缺失值处理
    for c in continuous_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in categorical_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0])

    # 异常值裁剪
    for col in continuous_cols:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower, upper)

    # one-hot
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 标准化
    scaler_used = scaler if scaler is not None else CustomScaler()
    if fit_scaler:
        X[continuous_cols] = scaler_used.fit_transform(X[continuous_cols])
    else:
        X[continuous_cols] = scaler_used.transform(X[continuous_cols])

    # 加入 bias
    X.insert(0, 'bias', 1.0)

    # 对齐列
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0.0)

    return X.values.astype(float), y, scaler_used, X.columns.tolist()


# --- 模型部分 ---

def normal_equation(X, y):
    try:
        return np.linalg.inv(X.T @ X) @ X.T @ y
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X.T @ X) @ X.T @ y


def compute_loss(X, y, theta):
    n = X.shape[0]
    errors = X @ theta - y
    mse = np.mean(errors ** 2)
    grad = (2.0 / n) * X.T @ errors
    return mse, grad


def batch_gradient_descent(X, y, lr=0.01, epochs=50, theta_init=None, verbose=False):
    n, m = X.shape
    theta = np.zeros((m, 1)) if theta_init is None else theta_init.copy()
    losses = []
    for epoch in range(epochs):
        mse, grad = compute_loss(X, y, theta)
        theta -= lr * grad
        losses.append(mse)
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f'BGD epoch {epoch}/{epochs}, mse={mse:.4f}')
    return theta, losses


def stochastic_gradient_descent(X, y, lr=0.01, epochs=100, theta_init=None, verbose=False):
    n, m = X.shape
    theta = np.zeros((m, 1)) if theta_init is None else theta_init.copy()
    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in perm:
            xi = X[i:i+1]
            yi = y[i:i+1]
            error = xi @ theta - yi
            grad = 2.0 * xi.T @ error
            theta -= lr * grad
        mse = np.mean((X @ theta - y) ** 2)
        losses.append(mse)
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f'SGD epoch {epoch}/{epochs}, mse={mse:.4f}')
    return theta, losses


def evaluate_model(X, y, theta):
    preds = X @ theta
    mse = custom_mean_squared_error(y, preds)
    mae = custom_mean_absolute_error(y, preds)
    return mse, mae


# --- 绘图函数（修改处）---

def plot_loss(losses, title, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='tab:blue', linewidth=2)
    plt.xlabel('Iterations / Epochs')
    plt.ylabel('MSE Loss')
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存：{save_path}")
    plt.show()


# --- 主函数 ---

def main():
    root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    csv_path = os.path.join(root, 'bike_sharing_hour.csv')
    df = load_data(csv_path)

    train_df, test_df = time_split(df)
    X_train, y_train, scaler, train_features = preprocess(train_df, fit_scaler=True)
    X_test, y_test, _, _ = preprocess(test_df, scaler=scaler, reference_columns=train_features)

    print(f"训练样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")
    print(f"测试样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]}")

    # 正规方程
    theta_ne = normal_equation(X_train, y_train)
    ne_train_mse, ne_train_mae = evaluate_model(X_train, y_train, theta_ne)
    ne_test_mse, ne_test_mae = evaluate_model(X_test, y_test, theta_ne)
    print(f"\n--- Normal Equation ---")
    print(f"Train MSE/MAE: {ne_train_mse:.4f} / {ne_train_mae:.4f}")
    print(f"Test  MSE/MAE: {ne_test_mse:.4f} / {ne_test_mae:.4f}")

    # BGD
    bgd_lr = 0.2
    theta_bgd, bgd_losses = batch_gradient_descent(X_train, y_train, lr=bgd_lr, epochs=50, verbose=True)
    bgd_train_mse, bgd_train_mae = evaluate_model(X_train, y_train, theta_bgd)
    bgd_test_mse, bgd_test_mae = evaluate_model(X_test, y_test, theta_bgd)
    print(f"\n--- Batch Gradient Descent ---")
    print(f"Train MSE/MAE: {bgd_train_mse:.4f} / {bgd_train_mae:.4f}")
    print(f"Test  MSE/MAE: {bgd_test_mse:.4f} / {bgd_test_mae:.4f}")

    # SGD
    sgd_lr = 0.001
    theta_sgd, sgd_losses = stochastic_gradient_descent(X_train, y_train, lr=sgd_lr, epochs=100, verbose=True)
    sgd_train_mse, sgd_train_mae = evaluate_model(X_train, y_train, theta_sgd)
    sgd_test_mse, sgd_test_mae = evaluate_model(X_test, y_test, theta_sgd)
    print(f"\n--- Stochastic Gradient Descent ---")
    print(f"Train MSE/MAE: {sgd_train_mse:.4f} / {sgd_train_mae:.4f}")
    print(f"Test  MSE/MAE: {sgd_test_mse:.4f} / {sgd_test_mae:.4f}")

    # --- 新绘图逻辑：分别绘制两张图 ---
    plot_loss(bgd_losses, title=f"BGD Loss Curve (lr={bgd_lr})",
              save_path=os.path.join(root, "BGD_Loss_Curve.png"))
    plot_loss(sgd_losses, title=f"SGD Loss Curve (lr={sgd_lr})",
              save_path=os.path.join(root, "SGD_Loss_Curve.png"))

    # 结果写入文件
    results_path = os.path.join(root, 'results_no_weekday.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write('--- Normal Equation ---\n')
        f.write(f'Train MSE/MAE: {ne_train_mse:.4f} / {ne_train_mae:.4f}\n')
        f.write(f'Test  MSE/MAE: {ne_test_mse:.4f} / {ne_test_mae:.4f}\n\n')

        f.write(f'--- Batch Gradient Descent (lr={bgd_lr}) ---\n')
        f.write(f'Train MSE/MAE: {bgd_train_mse:.4f} / {bgd_train_mae:.4f}\n')
        f.write(f'Test  MSE/MAE: {bgd_test_mse:.4f} / {bgd_test_mae:.4f}\n\n')

        f.write(f'--- Stochastic Gradient Descent (lr={sgd_lr}) ---\n')
        f.write(f'Train MSE/MAE: {sgd_train_mse:.4f} / {sgd_train_mae:.4f}\n')
        f.write(f'Test  MSE/MAE: {sgd_test_mse:.4f} / {sgd_test_mae:.4f}\n')

    print(f"\n结果已写入 {results_path}")


if __name__ == '__main__':
    main()
