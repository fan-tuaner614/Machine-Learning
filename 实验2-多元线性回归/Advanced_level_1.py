"""
目标：用环境因素预测共享单车每小时租赁量（目标列为 `cnt`）。

说明：
- 核心修改：手动实现 Ridge, LASSO, K-Fold 交叉验证。
- 依赖库：numpy, pandas, matplotlib, scipy (用于 P-值 和诊断)
- 任务：
    1. 手动实现 Ridge (正规方程) 和 LASSO (坐标下降)。
    2. 手动实现 K-Fold 交叉验证来选择最佳 alpha。
    3. 分析 LASSO 的特征筛选结果。
    4. 进行模型诊断（残差分析、正态性检验）。
    5. 可视化真实值 vs 预测值的时间序列。
"""

import os
import time  # 导入 time 库来计算运行时间
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# --- 支持中文显示 ---
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
matplotlib.rcParams['font.family'] = 'sans-serif'

import scipy.stats as sps


# --- 复用 3.py 中的辅助函数 ---

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
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform first.")
        return (data - self.mean_) / self.std_

def custom_mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def custom_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

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

def preprocess_with_interactions(df, scaler=None, fit_scaler=False, reference_columns=None):
    # (与 3.py 完全相同)
    continuous_cols = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_cols = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    y = df['cnt'].astype(float).to_numpy().reshape(-1, 1)

    features_needed = continuous_cols + categorical_cols + ['hr']
    X_df = df[features_needed].copy()
    X_df['time_period'] = X_df['hr'].apply(map_hour_to_3_periods)
    X_df = X_df.drop('hr', axis=1)
    categorical_cols.append('time_period')

    for c in continuous_cols:
        if X_df[c].isnull().any(): X_df[c] = X_df[c].fillna(X_df[c].median())
    for c in categorical_cols:
        if X_df[c].isnull().any(): X_df[c] = X_df[c].fillna(X_df[c].mode().iloc[0])

    for col in continuous_cols:
        lower = X_df[col].quantile(0.01)
        upper = X_df[col].quantile(0.99)
        X_df[col] = X_df[col].clip(lower, upper)

    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
        
    if 'holiday_1' in X_df.columns:
        X_df['temp_x_holiday'] = X_df['temp'] * X_df['holiday_1']
    if 'weathersit_3' in X_df.columns and 'time_period_peak' in X_df.columns:
         X_df['weathersit_x_peak'] = X_df['weathersit_3'] * X_df['time_period_peak']
    if 'workingday_1' in X_df.columns:
         X_df['hum_x_workingday'] = X_df['hum'] * X_df['workingday_1']

    scaler_used = scaler if scaler is not None else CustomScaler()
    cols_to_scale = [c for c in continuous_cols if c in X_df.columns]
    
    if cols_to_scale:
        if fit_scaler:
            X_df[cols_to_scale] = scaler_used.fit_transform(X_df[cols_to_scale])
        else:
            X_df[cols_to_scale] = scaler_used.transform(X_df[cols_to_scale])

    X_df.insert(0, 'bias', 1.0) 

    if reference_columns is not None:
        X_df = X_df.reindex(columns=reference_columns, fill_value=0.0)

    feature_names = X_df.columns.tolist()
    return X_df.values.astype(float), y, scaler_used, feature_names

def evaluate_model(X, y, theta):
    preds = X.dot(theta)
    mse = custom_mean_squared_error(y, preds)
    mae = custom_mean_absolute_error(y, preds)
    return mse, mae

def normal_equation(X, y):
    """ (复用) OLS 基线模型 """
    try:
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# --- 核心新增：手动实现 Ridge (岭回归) ---
def ridge_regression_normal(X, y, alpha):
    """
    使用正规方程手动实现岭回归。
    theta = (X.T * X + alpha * I')^(-1) * X.T * y
    """
    n, m = X.shape
    
    # 创建单位矩阵 I'
    I_prime = np.identity(m)
    # 不对 bias (j=0) 进行惩罚
    I_prime[0, 0] = 0.0 
    
    # (X.T * X + alpha * I')
    XTX = X.T.dot(X)
    penalty_term = alpha * I_prime
    
    try:
        # (X.T * X + alpha * I')^(-1)
        inv_term = np.linalg.inv(XTX + penalty_term)
    except np.linalg.LinAlgError:
        inv_term = np.linalg.pinv(XTX + penalty_term)
        
    # theta = inv * X.T * y
    theta = inv_term.dot(X.T).dot(y)
    return theta

# --- 核心新增：手动实现 LASSO (坐标下降) ---

def soft_thresholding(rho, lambda_):
    """
    软阈值操作符，LASSO 的核心。
    """
    if rho > lambda_:
        return rho - lambda_
    elif rho < -lambda_:
        return rho + lambda_
    else:
        return 0.0

def lasso_coordinate_descent(X, y, alpha, epochs=1000, tol=1e-5):
    """
    使用坐标下降法手动实现 LASSO 回归。
    
    损失函数: J(theta) = (1/n) * ||y - X*theta||^2 + alpha * ||theta||_1
    """
    n, m = X.shape
    
    # 扁平化 y
    y_flat = y.ravel()
    
    # 初始化 theta
    theta = np.zeros(m)
    
    # 预先计算 X_j 的 L2 范数平方 (用于归一化)
    # z_j = (1/n) * sum(X_ij^2) * 2  (来自 MSE 梯度的 2/n)
    z = (2.0 / n) * np.sum(X**2, axis=0)
    # 防止除以零
    z[z == 0] = 1.0 
    
    for epoch in range(epochs):
        theta_old = theta.copy()
        
        for j in range(m):
            # 计算不含 j 的预测
            y_pred = X.dot(theta)
            r_j = y_flat - y_pred + X[:, j] * theta[j] # 残差
            
            # rho_j = (1/n) * sum(X_ij * r_j) * 2
            rho_j = (2.0 / n) * np.dot(X[:, j], r_j)
            
            lambda_ = alpha # alpha 对应于 L1 惩罚项
            
            if j == 0:
                # bias 项，不惩罚
                theta[j] = rho_j / z[j]
            else:
                # 特征项，使用软阈值
                theta[j] = soft_thresholding(rho_j, lambda_) / z[j]
        
        # 检查收敛
        if np.linalg.norm(theta - theta_old) < tol:
            break
            
    # 返回 (m, 1) 形状的 theta
    return theta.reshape(-1, 1)

# --- 核心新增：手动实现 K-Fold 交叉验证 ---
def k_fold_cross_validation(model_type, X_train, y_train, alphas, k_folds=5):
    """
    手动实现 K-Fold 交叉验证来选择最佳 alpha。
    """
    n = X_train.shape[0]
    
    # 1. 打乱数据
    indices = np.arange(n)
    np.random.shuffle(indices)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    # 2. 计算 fold 大小
    fold_size = n // k_folds
    
    alpha_mses = {alpha: [] for alpha in alphas}
    
    print(f"--- K-Fold CV ({model_type}), k={k_folds} ---")
    
    # 3. 循环 k-folds
    for i in range(k_folds):
        # 4. 切分 训练集/验证集
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        
        val_indices = np.arange(val_start, val_end)
        train_indices = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
        
        X_sub_train = X_shuffled[train_indices]
        y_sub_train = y_shuffled[train_indices]
        X_val = X_shuffled[val_indices]
        y_val = y_shuffled[val_indices]
        
        # 5. 循环 alphas
        for alpha in alphas:
            try:
                if model_type == 'ridge':
                    theta = ridge_regression_normal(X_sub_train, y_sub_train, alpha)
                
                elif model_type == 'lasso':
                    # CV 时使用较少 epoch 加速
                    theta = lasso_coordinate_descent(X_sub_train, y_sub_train, alpha, epochs=500, tol=1e-3)
                
                # 6. 评估
                mse, _ = evaluate_model(X_val, y_val, theta)
                alpha_mses[alpha].append(mse)
                
            except Exception as e:
                print(f"Alpha {alpha} 在 fold {i} 失败: {e}")
                alpha_mses[alpha].append(np.inf) # 失败则设为无穷大
        
        print(f"Fold {i+1}/{k_folds} 完成。")

    # 7. 计算平均 MSE
    avg_mses = {alpha: np.mean(mses) for alpha, mses in alpha_mses.items() if np.isfinite(np.mean(mses))}
    
    if not avg_mses:
        print("CV 失败，所有 alpha 均未成功。")
        return None

    # 8. 找到最佳 alpha
    best_alpha = min(avg_mses, key=avg_mses.get)
    return best_alpha


# --- 复用 (上一个回答中)：LASSO 特征分析 ---
def analyze_lasso_features(lasso_theta, feature_names):
    """
    分析 LASSO 回归的系数，找出被筛选掉的特征。
    """
    print("\n" + "="*70)
    print("         LASSO 特征筛选分析")
    print("="*70)
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lasso_theta.flatten()
    })
    
    selected_features = coef_df[np.abs(coef_df['Coefficient']) > 1e-6] # 容忍度
    eliminated_features = coef_df[np.abs(coef_df['Coefficient']) <= 1e-6]

    print(f"总特征数: {len(feature_names)}")
    print(f"LASSO 选择的特征数: {len(selected_features)}")
    print(f"LASSO 淘汰的特征数: {len(eliminated_features)}")
    
    print("\n--- 被淘汰 (系数~=0) 的特征 ---")
    if len(eliminated_features) > 0:
        print(eliminated_features['Feature'].to_list())
    else:
        print("LASSO 没有淘汰任何特征 (可能 alpha 值很小)。")

    print("\n--- 被选中 (系数!=0) 的特征 (按绝对值排序) ---")
    selected_features['Abs_Coefficient'] = selected_features['Coefficient'].abs()
    print(selected_features.sort_values(by='Abs_Coefficient', ascending=False).to_string(index=False))
    print("="*70 + "\n")


# --- 复用 (上一个回答中)：模型诊断 ---
def perform_model_diagnostics(y_true, y_pred, model_name="Model"):
    """
    执行模型诊断 (依赖 scipy.stats)。
    """
    print("\n" + "="*70)
    print(f"         {model_name} 模型诊断")
    print("="*70)

    residuals = (y_true - y_pred).flatten()
    y_pred = y_pred.flatten()
    
    # 1. 残差图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('预测值 (Predicted Values)')
    plt.ylabel('残差 (Residuals)')
    plt.title('残差图 (Residuals vs. Fitted)')

    # 2. Q-Q 图
    plt.subplot(1, 2, 2)
    sps.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差正态性 Q-Q 图')
    
    plt.tight_layout()
    plt.savefig(f'diagnostics_{model_name}.png')
    print(f"诊断图已保存为: diagnostics_{model_name}.png")
    
    # 3. Shapiro-Wilk 正态性检验
    if len(residuals) > 5000:
        residuals_sample = np.random.choice(residuals, 5000, replace=False)
    else:
        residuals_sample = residuals
        
    try:
        shapiro_stat, shapiro_p = sps.shapiro(residuals_sample)
        print("\n--- 残差正态性检验 (Shapiro-Wilk Test) ---")
        print(f"  T-Statistic: {shapiro_stat:.4f}")
        print(f"  P-Value: {shapiro_p:.4e}")
        if shapiro_p < 0.05:
            print("  -> 结论: P < 0.05，残差不符合正态分布。")
        else:
            print("  -> 结论: P > 0.05，无法拒绝残差呈正态分布的假设。")
    except Exception as e:
        print(f"\n无法执行 Shapiro-Wilk 检验: {e}")
    print("="*70 + "\n")


# --- 复用 (上一个回答中)：时间序列可视化 ---
def plot_time_series(y_true, y_pred, model_name="Model", sample_size=500):
    """
    绘制真实值 vs 预测值的时间序列图（抽样）。
    """
    print("\n" + "="*70)
    print(f"         {model_name} 时间序列预测效果")
    print("="*70)
    
    y_true_sample = y_true.flatten()[:sample_size]
    y_pred_sample = y_pred.flatten()[:sample_size]
    time_index = np.arange(len(y_true_sample))
    
    plt.figure(figsize=(15, 7))
    plt.plot(time_index, y_true_sample, label='真实值 (Actual)', color='blue', alpha=0.9)
    plt.plot(time_index, y_pred_sample, label='预测值 (Predicted)', color='red', linestyle='--', alpha=0.8)
    plt.xlabel(f'时间点 (测试集前 {sample_size} 小时)')
    plt.ylabel('单车租赁量 (cnt)')
    plt.title(f'{model_name}: 真实值 vs 预测值 (时间序列对比)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'timeseries_{model_name}.png')
    print(f"时间序列图已保存为: timeseries_{model_name}.png")
    print("="*70 + "\n")


# --- 主函数 (修改版) ---
def main():
    root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    csv_path = os.path.join(root, 'bike_sharing_hour.csv')
    df = load_data(csv_path)

    train_df, test_df = time_split(df, train_frac=0.7)

    X_train, y_train, scaler, train_feature_names = preprocess_with_interactions(
        train_df, scaler=None, fit_scaler=True
    )
    X_test, y_test, _, _ = preprocess_with_interactions(
        test_df, scaler=scaler, fit_scaler=False, reference_columns=train_feature_names
    )

    print(f'训练样本数: {X_train.shape[0]}, 特征数 (含交互): {X_train.shape[1]}')
    
    # --- 1. OLS (正规方程) 基线模型 ---
    print("\n--- 正在训练 OLS (基线) ---")
    start_time = time.time()
    theta_ne = normal_equation(X_train, y_train)
    ols_train_mse, ols_train_mae = evaluate_model(X_train, y_train, theta_ne)
    ols_test_mse, ols_test_mae = evaluate_model(X_test, y_test, theta_ne)
    print(f"OLS 耗时: {time.time() - start_time:.2f} 秒")
    
    print(f'\n--- OLS (基线) ---')
    print(f'Train MSE/MAE: {ols_train_mse:.4f}, {ols_train_mae:.4f}')
    print(f'Test  MSE/MAE: {ols_test_mse:.4f}, {ols_test_mae:.4f}')

    # --- 2. 交叉验证选择 Alpha ---
    # (注意：手动 CV 很慢，我们减少 alpha 的数量)
    alphas_to_try = np.logspace(-5, 1, 15) # 15 个点
    
    # Ridge CV
    print("\n--- 正在为 岭回归 (Ridge) 执行交叉验证 ---")
    start_time = time.time()
    best_alpha_ridge = k_fold_cross_validation(
        'ridge', X_train, y_train, alphas_to_try, k_folds=5
    )
    print(f"Ridge CV 耗时: {time.time() - start_time:.2f} 秒")
    print(f"Ridge 选中的最佳 Alpha: {best_alpha_ridge:.6f}")

    # LASSO CV
    print("\n--- 正在为 LASSO 执行交叉验证 (这可能需要几分钟) ---")
    start_time = time.time()
    best_alpha_lasso = k_fold_cross_validation(
        'lasso', X_train, y_train, alphas_to_try, k_folds=5
    )
    print(f"LASSO CV 耗时: {time.time() - start_time:.2f} 秒")
    print(f"LASSO 选中的最佳 Alpha: {best_alpha_lasso:.6f}")

    # --- 3. 训练和评估最终模型 ---
    
    # 训练最终 Ridge
    print("\n--- 正在训练最终 Ridge 模型 ---")
    theta_ridge = ridge_regression_normal(X_train, y_train, best_alpha_ridge)
    ridge_train_mse, ridge_train_mae = evaluate_model(X_train, y_train, theta_ridge)
    ridge_test_mse, ridge_test_mae = evaluate_model(X_test, y_test, theta_ridge)
    
    print(f'\n--- Ridge (CV, alpha={best_alpha_ridge:.6f}) ---')
    print(f'Train MSE/MAE: {ridge_train_mse:.4f}, {ridge_train_mae:.4f}')
    print(f'Test  MSE/MAE: {ridge_test_mse:.4f}, {ridge_test_mae:.4f}')

    # 训练最终 LASSO (使用更多 epochs)
    print("\n--- 正在训练最终 LASSO 模型 (这可能需要一点时间) ---")
    start_time = time.time()
    theta_lasso = lasso_coordinate_descent(X_train, y_train, best_alpha_lasso, epochs=2000, tol=1e-6)
    print(f"LASSO 训练耗时: {time.time() - start_time:.2f} 秒")
    lasso_train_mse, lasso_train_mae = evaluate_model(X_train, y_train, theta_lasso)
    lasso_test_mse, lasso_test_mae = evaluate_model(X_test, y_test, theta_lasso)
    
    print(f'\n--- LASSO (CV, alpha={best_alpha_lasso:.6f}) ---')
    print(f'Train MSE/MAE: {lasso_train_mse:.4f}, {lasso_train_mae:.4f}')
    print(f'Test  MSE/MAE: {lasso_test_mse:.4f}, {lasso_test_mae:.4f}')

    # --- 4. LASSO 特征分析 ---
    analyze_lasso_features(theta_lasso, train_feature_names)
    
    # --- 5. 模型诊断 (以 LASSO 为例) ---
    y_pred_test_lasso = X_test.dot(theta_lasso)
    perform_model_diagnostics(y_test, y_pred_test_lasso, model_name="LASSO_Test_Manual")

    # --- 6. 时间序列可视化 (以 LASSO 为例) ---
    plot_time_series(y_test, y_pred_test_lasso, model_name="LASSO_Test_Manual", sample_size=500)


if __name__ == '__main__':
    main()