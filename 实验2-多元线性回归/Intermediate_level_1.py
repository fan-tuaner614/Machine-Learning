"""
目标：用环境因素预测共享单车每小时租赁量（目标列为 `cnt`）。

说明：
- 核心修改：移除 'weekday' (星期几) 特征。继续使用三级时段划分法处理小时 'hr'。
- 本代码不依赖 scikit-learn 库，所有功能均使用 numpy 和 pandas 实现。
- 连续（数值）特征：`temp`, `atemp`, `hum`, `windspeed`
- 分类特征（应做 one-hot）：`season`, `yr`, `holiday`, `workingday`, `weathersit`, 以及新创建的 `time_period`
- 目标列（标签）：`cnt`

使用：在包含 `bike_sharing_hour.csv` 的目录运行：
    python bike_predictor_no_weekday.py

依赖：numpy, pandas, matplotlib
安装：pip install numpy pandas matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 自定义实现，替代 scikit-learn ---

class CustomScaler:
    """
    功能与 scikit-learn 的 StandardScaler 相同的自定义类。
    用于对数据进行标准化处理 Z = (X - μ) / σ。
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, data):
        """
        计算训练数据的均值和标准差，并用其进行标准化。
        """
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # 防止除以零
        self.std_[self.std_ == 0] = 1.0
        return (data - self.mean_) / self.std_

    def transform(self, data):
        """
        使用已存储的均值和标准差（来自训练集）来标准化新数据（测试集）。
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform first.")
        return (data - self.mean_) / self.std_

def custom_mean_squared_error(y_true, y_pred):
    """自定义实现的均方误差 (MSE)。"""
    return np.mean((y_pred - y_true) ** 2)

def custom_mean_absolute_error(y_true, y_pred):
    """自定义实现的平均绝对误差 (MAE)。"""
    return np.mean(np.abs(y_pred - y_true))


# --- 数据处理与模型 ---

def load_data(path):
    """按时间顺序读取 CSV 数据并返回 DataFrame。"""
    df = pd.read_csv(path)
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)
    return df


def time_split(df, train_frac=0.7):
    """按时间顺序切分数据集，避免数据泄漏。"""
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test


def map_hour_to_3_periods(hour):
    """
    按“三级划分法”将小时映射到指定的时间段。
    """
    if hour in [7, 8, 9, 16, 17, 18, 19]:
        return 'peak'
    elif hour in [10, 11, 12, 13, 14, 15, 20]:
        return 'off_peak'
    else:
        return 'low_hours'


def preprocess(df, scaler=None, fit_scaler=False, reference_columns=None):
    """
    对原始 DataFrame 做预处理并返回特征矩阵 X 与目标 y。
    核心改动：移除了 'weekday' 特征。
    """
    # 1. 定义连续和分类特征
    continuous_cols = ['temp', 'atemp', 'hum', 'windspeed']
    # 从分类特征列表中移除 'weekday'
    categorical_cols = ['season', 'yr', 'holiday', 'workingday', 'weathersit']

    y = df['cnt'].astype(float).to_numpy().reshape(-1, 1)

    # 2. 选取特征，并创建新的 'time_period' 特征
    features_needed = continuous_cols + categorical_cols + ['hr']
    X = df[features_needed].copy()
    
    X['time_period'] = X['hr'].apply(map_hour_to_3_periods)
    X = X.drop('hr', axis=1)
    
    categorical_cols.append('time_period')

    # 3. 缺失值处理
    for c in continuous_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in categorical_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].mode().iloc[0])

    # 4. 异常值处理 (裁剪)
    for col in continuous_cols:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower, upper)

    # 5. 对分类列进行 one-hot 编码
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 6. 标准化连续特征
    scaler_used = scaler if scaler is not None else CustomScaler()
    cols_to_scale = [c for c in continuous_cols if c in X.columns]
    if cols_to_scale:
        if fit_scaler:
            X[cols_to_scale] = scaler_used.fit_transform(X[cols_to_scale])
        else:
            X[cols_to_scale] = scaler_used.transform(X[cols_to_scale])

    # 7. 添加常数项 (bias)
    X.insert(0, 'bias', 1.0)

    # 8. 对齐测试集和训练集的列
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0.0)

    feature_names = X.columns.tolist()
    return X.values.astype(float), y, scaler_used, feature_names


def normal_equation(X, y):
    """使用正规方程（闭式解）计算线性回归参数。"""
    try:
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def compute_loss(X, y, theta):
    """计算 MSE 损失和梯度。"""
    n = X.shape[0]
    errors = X.dot(theta) - y
    mse = np.mean(errors ** 2)
    grad = (2.0 / n) * X.T.dot(errors)
    return mse, grad


def batch_gradient_descent(X, y, lr=0.01, epochs=50, theta_init=None, verbose=False):
    """批量梯度下降（BGD）。"""
    n, m = X.shape
    theta = np.zeros((m, 1)) if theta_init is None else theta_init.copy()
    losses = []
    for epoch in range(epochs):
        mse, grad = compute_loss(X, y, theta)
        theta -= lr * grad
        losses.append(mse)
        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f'BGD epoch {epoch}/{epochs} mse={mse:.4f}')
    return theta, losses


def stochastic_gradient_descent(X, y, lr=0.01, epochs=100, theta_init=None, verbose=False):
    """随机梯度下降（SGD）。"""
    n, m = X.shape
    theta = np.zeros((m, 1)) if theta_init is None else theta_init.copy()
    epoch_losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]
        for i in range(n):
            xi = X_shuf[i:i+1]
            yi = y_shuf[i:i+1]
            error = xi.dot(theta) - yi
            grad = 2.0 * xi.T.dot(error)
            theta -= lr * grad
        
        mse = np.mean((X.dot(theta) - y) ** 2)
        epoch_losses.append(mse)
        if verbose:
            print(f'SGD epoch {epoch}/{epochs} mse={mse:.4f}')
    return theta, epoch_losses


def evaluate_model(X, y, theta):
    """评估模型性能，返回 MSE 和 MAE。"""
    preds = X.dot(theta)
    mse = custom_mean_squared_error(y, preds)
    mae = custom_mean_absolute_error(y, preds)
    return mse, mae


def plot_losses(loss_dict, title='Loss curves'):
    """绘制损失曲线。"""
    plt.figure(figsize=(10, 6))
    for label, losses in loss_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel('Iterations / Epochs')
    plt.ylabel('MSE')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def analyze_coefficients(theta, feature_names):
    """
    分析模型的系数(theta)，提供特征重要性排序和业务解读。
    """
    print("\n" + "="*40)
    print("         特征重要性与业务解读")
    print("="*40)
    
    if theta.ndim > 1:
        theta_flat = theta.flatten()
    else:
        theta_flat = theta
        
    if len(theta_flat) != len(feature_names):
        print(f"错误: 系数(theta)数量 {len(theta_flat)} 与特征名称数量 {len(feature_names)} 不匹配。")
        return

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': theta_flat
    })
    
    # 按系数绝对值排序，找出“最重要”的特征
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
    
    # 过滤掉 bias (常数项)
    coef_df_no_bias = coef_df[coef_df['Feature'].str.lower() != 'bias']
    
    print("--- 特征重要性排序 (按系数绝对值) ---")
    print(coef_df_no_bias.head(10).to_string(index=False))
    
    print("\n--- 关键特征业务解读 ---")
    
    # 用于解读的辅助函数
    def interpret(feature_name, positive_desc, negative_desc):
        if feature_name not in coef_df['Feature'].values:
            # 可能是因为 one-hot 编码中的 drop_first 导致该特征是基准
            # print(f" (特征 '{feature_name}' 未在模型中找到, 可能作为基准)")
            return
        
        coef = coef_df[coef_df['Feature'] == feature_name]['Coefficient'].values[0]
        
        if coef > 0.1:
            print(f"  - {feature_name}: 系数 {coef:+.2f} (正相关)。")
            print(f"    -> 解读: {positive_desc}")
        elif coef < -0.1:
            print(f"  - {feature_name}: 系数 {coef:+.2f} (负相关)。")
            print(f"    -> 解读: {negative_desc}")
        else:
            print(f"  - {feature_name}: 系数 {coef:+.2f} (影响微弱)。")

    # 注意：这些解读基于 one-hot 编码的基准 (drop_first=True)
    # season (基准: 1-春天), yr (基准: 0-2011), weathersit (基准: 1-晴天), time_period (基准: low_hours)
    
    print("\n[环境因素]")
    interpret('temp', '温度越高，租赁量越多。', '温度越低，租赁量越少。')
    interpret('hum', '湿度越高，租赁量越多。', '湿度越低，租赁量越少。') # 注意：这个可能与直觉相反，看数据
    interpret('windspeed', '风速越大，租赁量越多。', '风速越低，租赁量越少。') # 注意：这个也可能与直觉相反
    interpret('weathersit_2', '天气为薄雾/阴天时，租赁量增加。', '天气为薄雾/阴天时，租赁量减少。')
    interpret('weathersit_3', '天气为小雨/小雪时，租赁量显著减少。', '天气为小雨/小雪时，租赁量增加 (不太可能)。')
    
    print("\n[时间因素]")
    interpret('yr_1', '2012年(yr=1)的租赁量显著高于2011年(基准)。', '2012年租赁量更低。')
    interpret('time_period_peak', '高峰时段(peak)的租赁量显著高于非高峰时段(low_hours)。', '高峰时段租赁量更低。')
    interpret('time_period_off_peak', '普通时段(off_peak)的租赁量高于非高峰时段(low_hours)。', '普通时段租赁量更低。')
    interpret('season_2', '夏天(season=2)的租赁量高于春天(基准)。', '夏天租赁量更低。')
    interpret('season_3', '秋天(season=3)的租赁量高于春天(基准)。', '秋天租赁量更低。')
    interpret('season_4', '冬天(season=4)的租赁量高于春天(基准)。', '冬天租赁量更低。')

    print("\n[其他因素]")
    interpret('holiday_1', '节假日(holiday=1)的租赁量更低 (可能因为通勤需求减少)。', '节假日租赁量更高。')
    interpret('workingday_1', '工作日(workingday=1)的租赁量更高 (可能因为通勤)。', '工作日租赁量更低。')
    
    print("="*40 + "\n")

def main():
    root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    csv_path = os.path.join(root, 'bike_sharing_hour.csv')
    df = load_data(csv_path)

    train_df, test_df = time_split(df, train_frac=0.7)

    X_train, y_train, scaler, train_feature_names = preprocess(train_df, scaler=None, fit_scaler=True)
    X_test, y_test, _, _ = preprocess(test_df, scaler=scaler, fit_scaler=False, reference_columns=train_feature_names)

    print(f'训练样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}')
    print(f'测试样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]}')

    # 正规方程基线
    theta_ne = normal_equation(X_train, y_train)
    ne_train_mse, ne_train_mae = evaluate_model(X_train, y_train, theta_ne)
    ne_test_mse, ne_test_mae = evaluate_model(X_test, y_test, theta_ne)
    print(f'\n--- Normal Equation ---')
    print(f'Train MSE/MAE: {ne_train_mse:.4f}, {ne_train_mae:.4f}')
    print(f'Test  MSE/MAE: {ne_test_mse:.4f}, {ne_test_mae:.4f}')

    analyze_coefficients(theta_ne, train_feature_names)

    # BGD 和 SGD 训练
    bgd_lr = 0.2
    sgd_lr = 0.001
    
    theta_bgd, bgd_losses = batch_gradient_descent(X_train, y_train, lr=bgd_lr, epochs=50, verbose=True)
    bgd_train_mse, bgd_train_mae = evaluate_model(X_train, y_train, theta_bgd)
    bgd_test_mse, bgd_test_mae = evaluate_model(X_test, y_test, theta_bgd)

    theta_sgd, sgd_losses = stochastic_gradient_descent(X_train, y_train, lr=sgd_lr, epochs=100, verbose=True)
    sgd_train_mse, sgd_train_mae = evaluate_model(X_train, y_train, theta_sgd)
    sgd_test_mse, sgd_test_mae = evaluate_model(X_test, y_test, theta_sgd)
    
    print(f'\n--- Batch Gradient Descent (lr={bgd_lr}) ---')
    print(f'Final Train MSE/MAE: {bgd_train_mse:.4f}, {bgd_train_mae:.4f}')
    print(f'Final Test  MSE/MAE: {bgd_test_mse:.4f}, {bgd_test_mae:.4f}')
    
    print(f'\n--- Stochastic Gradient Descent (lr={sgd_lr}) ---')
    print(f'Final Train MSE/MAE: {sgd_train_mse:.4f}, {sgd_train_mae:.4f}')
    print(f'Final Test  MSE/MAE: {sgd_test_mse:.4f}, {sgd_test_mae:.4f}')

    # 绘制 BGD/SGD 收敛曲线
    plot_losses({f'BGD lr={bgd_lr}': bgd_losses, f'SGD lr={sgd_lr}': sgd_losses}, title='BGD vs SGD Loss Convergence')
    
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
    print(f'\n结果已写入 {results_path}')




if __name__ == '__main__':
    main()

# [环境因素]

# temp: 系数 +25.58 (正相关)。解读：温度越高，租赁量越多。 (正确)

# hum: 系数 -21.36 (负相关)。

# 错误解读: 湿度越低，租赁量越少。

# 正确解读: 湿度越高，租赁量越少。 (这很合理，天气太潮湿或闷热，骑车的人会减少)。

# windspeed: 系数 -3.50 (负相关)。

# 错误解读: 风速越低，租赁量越少。

# 正确解读: 风速越高，租赁量越少。 (风太大不适合骑车，但这个系数很小，说明影响不大)。

# weathersit_2 (薄雾/阴天): 系数 -7.91 (负相关)。解读：天气为薄雾/阴天时，租赁量减少。(正确)。

# weathersit_3 (小雨/小雪): 系数 -48.08 (负相关)。

# 错误解读: 租赁量增加 (不太可能)。

# 正确解读: 天气为小雨/小雪时，租赁量显著减少。 (这非常符合直觉)。

# [时间因素]

# yr_1: 系数 +79.30 (正相关)。解读：2012年(yr=1)的租赁量显著高于2011年(基准)。(正确)。这说明业务在2012年有显著增长。

# time_period_peak: 系数 +204.54 (正相关)。解读：高峰时段(peak)的租赁量显著高于非高峰时段(low_hours)。(正确)。这是通勤需求。

# time_period_off_peak: 系数 +106.96 (正相关)。解读：普通时段(off_peak)的租赁量高于非高峰时段(low_hours)。(正确)。

# season_2 (夏天): 系数 +27.35 (正相关)。解读：夏天(season=2)的租赁量高于春天(基准)。(正确)。

# season_3 (秋天): 系数 +0.04 (影响微弱)。(正确)。说明秋天和春天(基准)的租赁量差不多。

# season_4 (冬天): 系数 +53.59 (正相关)。解读：冬天(season=4)的租赁量高于春天(基准)。

# 注意: 这个结果虽然系数为正，但可能有点反直觉（冬天骑车的人比春天还多？）。但这可能是因为模型同时考虑了 temp (温度)。春天(season=1)可能因为多雨或气温回暖不稳定，导致骑车的人反而不如（刨除温度影响后的）冬天。

# [其他因素]

# holiday_1: 系数 -22.28 (负相关)。

# 错误解读: 节假日租赁量更高。

# 正确解读: 节假日租赁量更低。 (这很合理，因为节假日大家不上班，导致 time_period_peak 的通勤高峰消失了)。

# workingday_1: 系数 +0.25 (正相关)。解读：工作日(workingday=1)的租赁量更高。

# 注意: 这个系数非常小（影响微弱）。这可能是因为time_period_peak (高峰时段) 已经把“工作日通勤”这个信息几乎完全吸收了，所以 workingday_1 变量本身不再提供太多额外信息。