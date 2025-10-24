"""
目标：用环境因素预测共享单车每小时租赁量（目标列为 `cnt`）。

说明：
- 核心修改：不使用 statsmodels，手动实现统计显著性 (P-值) 和交互特征。
- 依赖库：numpy, pandas, matplotlib, scipy (仅用于 P-值计算)
- 交互特征 (更新版)：
    1. temp * holiday_1 (温度与假期的交互)
    2. weathersit_3 * time_period_peak (坏天气与高峰期的交互)
    3. hum * workingday_1 (湿度与工作日的交互)
- 统计分析：
    - 手动计算标准误 (SE), t-统计量, P-值。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps  # *** 新增：仅用于 t-分布的 P-值计算 ***

# --- 自定义实现 (与 1.py 相同) ---

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

# --- 数据处理 (修改版：更新交互特征) ---

def preprocess_with_interactions(df, scaler=None, fit_scaler=False, reference_columns=None):
    """
    对原始 DataFrame 做预处理，并手动添加交互特征。
    返回 X (DataFrame) 和 y (Numpy Array)
    """
    continuous_cols = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_cols = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    y = df['cnt'].astype(float).to_numpy().reshape(-1, 1)

    features_needed = continuous_cols + categorical_cols + ['hr']
    X_df = df[features_needed].copy()
    X_df['time_period'] = X_df['hr'].apply(map_hour_to_3_periods)
    X_df = X_df.drop('hr', axis=1)
    categorical_cols.append('time_period')

    # 缺失值处理
    for c in continuous_cols:
        if X_df[c].isnull().any(): X_df[c] = X_df[c].fillna(X_df[c].median())
    for c in categorical_cols:
        if X_df[c].isnull().any(): X_df[c] = X_df[c].fillna(X_df[c].mode().iloc[0])

    # 异常值处理
    for col in continuous_cols:
        lower = X_df[col].quantile(0.01)
        upper = X_df[col].quantile(0.99)
        X_df[col] = X_df[col].clip(lower, upper)

    # *** 在创建交互项之前进行 One-hot 编码 ***
    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)

    # --- 核心修改：创建新的三组交互特征 ---
    
    # 1. 温度 x 假期 (假期的休闲用户是否对温度更敏感？)
    if 'holiday_1' in X_df.columns:
        X_df['temp_x_holiday'] = X_df['temp'] * X_df['holiday_1']

    # 2. 坏天气 x 高峰期 (高峰时段下雨/雪是否导致需求锐减？)
    if 'weathersit_3' in X_df.columns and 'time_period_peak' in X_df.columns:
         X_df['weathersit_x_peak'] = X_df['weathersit_3'] * X_df['time_period_peak']
         
    # 3. 湿度 x 工作日 (工作日的通勤者是否对湿度容忍度更高？)
    if 'workingday_1' in X_df.columns:
         # 注意: hum 是标准化过的, workingday_1 是 0/1
         X_df['hum_x_workingday'] = X_df['hum'] * X_df['workingday_1']
         
    # --- 交互结束 ---

    # 标准化连续特征
    scaler_used = scaler if scaler is not None else CustomScaler()
    cols_to_scale = [c for c in continuous_cols if c in X_df.columns]
    
    if cols_to_scale:
        if fit_scaler:
            X_df[cols_to_scale] = scaler_used.fit_transform(X_df[cols_to_scale])
        else:
            X_df[cols_to_scale] = scaler_used.transform(X_df[cols_to_scale])

    # 添加常数项 (bias)
    X_df.insert(0, 'bias', 1.0)

    if reference_columns is not None:
        X_df = X_df.reindex(columns=reference_columns, fill_value=0.0)

    feature_names = X_df.columns.tolist()
    return X_df.values.astype(float), y, scaler_used, feature_names


# --- 模型 (修改版：正规方程返回P值计算所需矩阵) ---

def normal_equation_with_stats(X, y):
    """
    使用正规方程计算参数，并返回 P-值 计算所需的 (X.T * X) 的逆矩阵。
    """
    try:
        XTX = X.T.dot(X)
        inv_XTX = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        XTX = X.T.dot(X)
        inv_XTX = np.linalg.pinv(XTX)
        
    theta = inv_XTX.dot(X.T).dot(y)
    return theta, inv_XTX


def evaluate_model(X, y, theta):
    """评估模型性能，返回 MSE 和 MAE。"""
    preds = X.dot(theta)
    mse = custom_mean_squared_error(y, preds)
    mae = custom_mean_absolute_error(y, preds)
    return mse, mae

# --- 核心新增：统计分析函数 (纯 Numpy + Scipy) ---

def analyze_coefficients_numpy(theta, inv_XTX, X_train, y_train, feature_names):
    """
    手动计算系数的 标准误、t-统计量 和 P-值。
    不使用 statsmodels。
    """
    print("\n" + "="*70)
    print("         手动统计分析 (P-值、重要性、交互项)")
    print("="*70)

    # 1. 计算基本参数
    n = X_train.shape[0]        # 样本量
    k = X_train.shape[1]        # 特征数 (含bias)
    dof = n - k                 # 自由度
    
    # 2. 计算残差
    y_pred = X_train.dot(theta)
    residuals = y_train - y_pred
    
    # 3. 计算残差平方和 (RSS)
    rss = residuals.T.dot(residuals)[0, 0] # 得到一个标量
    
    # 4. 估计误差方差
    sigma_squared_hat = rss / dof
    
    # 5. 计算系数的方差-协方差矩阵
    var_cov_matrix = sigma_squared_hat * inv_XTX
    
    # 6. 提取标准误 (SE)
    # 标准误是方差的平方根，位于方差-协方差矩阵的对角线上
    coef_variances = np.diag(var_cov_matrix)
    std_errors = np.sqrt(coef_variances)
    
    # 7. 计算 t-统计量
    theta_flat = theta.flatten()
    t_stats = theta_flat / std_errors
    
    # 8. 计算 P-值
    # 使用 scipy.stats.t 的 sf (Survival Function)
    # p_value = 2 * (1 - CDF(|t|)) = 2 * SF(|t|)
    p_values = 2 * sps.t.sf(np.abs(t_stats), df=dof)
    
    # --- 结果汇总与展示 ---
    
    # 汇总到 DataFrame
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': theta_flat,
        'Std_Error': std_errors,
        't_Statistic': t_stats,
        'P_Value': p_values
    })
    
    results_df['Abs_Coefficient'] = results_df['Coefficient'].abs()
    
    # 添加显著性标记
    def get_significance(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''
        
    results_df['Signif'] = results_df['P_Value'].apply(get_significance)

    print("\n--- 统计显著性分析 (P-值 < 0.05 则显著) ---")
    print("Signif codes: 0 '***' 0.001 '**' 0.01 '*' 0.05")
    # 设置 pandas 打印格式
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
    print(results_df.sort_values(by='Abs_Coefficient', ascending=False).to_string(index=False))
    
    
    print("\n--- 业务解读与交互分析 ---")
    
    # 辅助函数
    def interpret_stat(feature_name):
        if feature_name not in results_df['Feature'].values:
            print(f"  - 特征 '{feature_name}' 未找到 (可能作为基准)。")
            return
        
        row = results_df[results_df['Feature'] == feature_name].iloc[0]
        coef = row['Coefficient']
        p_val = row['P_Value']
        
        if p_val < 0.05:
            if coef > 0:
                print(f"  - {feature_name}: 显著正相关 (Coef: {coef:+.2f}, P: {p_val:.3e})")
                
                # *** 新增/修改的交互解读 ***
                if 'temp_x_holiday' in feature_name:
                    print("    -> 解读: 节假日时，温度升高的正面效应被放大了 (休闲用户更喜欢好天气)。")
                elif 'weathersit_x_peak' in feature_name:
                    print("    -> 解读: 高峰时段的坏天气反而导致需求增加 (不符合直觉，需检查)。")
                elif 'hum_x_workingday' in feature_name:
                    print("    -> 解读: 湿度对工作日有额外的正向影响 (可能通勤者对湿度的容忍度高于非通勤者)。")
                # *** 结束 ***

                elif 'temp' in feature_name:
                    print("    -> 解读: 温度越高，租赁量越多。")
                elif 'yr_1' in feature_name:
                    print("    -> 解读: 2012年(yr=1)的租赁量显著高于2011年(基准)。")
                elif 'time_period_peak' in feature_name:
                    print("    -> 解读: 高峰时段(peak)的租赁量显著高于非高峰时段(low_hours)。")
            else:
                print(f"  - {feature_name}: 显著负相关 (Coef: {coef:+.2f}, P: {p_val:.3e})")
                
                # *** 新增/修改的交互解读 ***
                if 'temp_x_holiday' in feature_name:
                    print("    -> 解读: 节假日时，温度升高的正面效应被削弱了 (不符合直觉)。")
                elif 'weathersit_x_peak' in feature_name:
                    print("    -> 解读: 高峰时段的坏天气(小雨/雪)导致租赁量额外大幅减少 (符合直觉，通勤者换乘)。")
                elif 'hum_x_workingday' in feature_name:
                    print("    -> 解读: 湿度对工作日有额外的负向影响 (可能通勤者比非通勤者更厌恶潮湿)。")
                # *** 结束 ***
                
                elif 'weathersit_3' in feature_name:
                    print("    -> 解读: 天气为小雨/小雪时，租赁量显著减少。")
                elif 'holiday_1' in feature_name:
                    print("    -> 解读: 节假日(holiday=1)的租赁量更低 (通勤需求减少)。")
        else:
            print(f"  - {feature_name}: 统计上不显著 (Coef: {coef:+.2f}, P: {p_val:.3f})")

    # 解读关键特征和交互特征
    print("\n[主效应]")
    interpret_stat('temp')
    interpret_stat('hum')
    interpret_stat('weathersit_3')
    interpret_stat('yr_1')
    interpret_stat('time_period_peak')
    interpret_stat('holiday_1')
    interpret_stat('workingday_1')


    print("\n[交互效应]")
    # *** 更新调用的特征名 ***
    interpret_stat('temp_x_holiday')
    interpret_stat('weathersit_x_peak')
    interpret_stat('hum_x_workingday')
    
    print("="*70 + "\n")
    # 恢复 pandas 默认打印
    pd.reset_option('display.float_format')


# --- 主函数 ---
def main():
    root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    csv_path = os.path.join(root, 'bike_sharing_hour.csv')
    df = load_data(csv_path)

    train_df, test_df = time_split(df, train_frac=0.7)

    # *** 使用新的预处理函数 ***
    X_train, y_train, scaler, train_feature_names = preprocess_with_interactions(
        train_df, scaler=None, fit_scaler=True
    )
    X_test, y_test, _, _ = preprocess_with_interactions(
        test_df, scaler=scaler, fit_scaler=False, reference_columns=train_feature_names
    )

    print(f'训练样本数: {X_train.shape[0]}, 特征数 (含交互): {X_train.shape[1]}')
    print(f'测试样本数: {X_test.shape[0]}, 特征数 (含交互): {X_test.shape[1]}')

    # *** 使用新的正规方程函数 ***
    theta_ne, inv_XTX = normal_equation_with_stats(X_train, y_train)
    ne_train_mse, ne_train_mae = evaluate_model(X_train, y_train, theta_ne)
    ne_test_mse, ne_test_mae = evaluate_model(X_test, y_test, theta_ne)
    
    print(f'\n--- Normal Equation 性能 (含交互特征) ---')
    print(f'Train MSE/MAE: {ne_train_mse:.4f}, {ne_train_mae:.4f}')
    print(f'Test  MSE/MAE: {ne_test_mse:.4f}, {ne_test_mae:.4f}')

    # *** 运行新增的统计分析 ***
    analyze_coefficients_numpy(theta_ne, inv_XTX, X_train, y_train, train_feature_names)

    # (BGD 和 SGD 部分被省略，因为它们不用于统计推断)
    # (如果需要，它们也可以运行，但 `analyze_coefficients_numpy` 仅适用于 OLS/正规方程)

if __name__ == '__main__':
    main()



# --- 统计显著性分析 (P-值 < 0.05 则显著) ---
# Signif codes: 0 '***' 0.001 '**' 0.01 '*' 0.05
#              Feature  Coefficient  Std_Error  t_Statistic  P_Value  Abs_Coefficient Signif
#     time_period_peak     210.4228     2.2898      91.8970   0.0000         210.4228    ***
# time_period_off_peak     107.4023     2.3176      46.3416   0.0000         107.4023    ***
#     hum_x_workingday      86.4730     9.5857       9.0210   0.0000          86.4730    ***
#                 yr_1      79.2203     2.2040      35.9447   0.0000          79.2203    ***
#       temp_x_holiday      77.8944    26.2800       2.9640   0.0030          77.8944     **
#    weathersit_x_peak     -58.5491     6.5430      -8.9483   0.0000          58.5491    ***
#            holiday_1     -58.5362    12.7046      -4.6075   0.0000          58.5362    ***
#         workingday_1     -53.8438     6.3024      -8.5434   0.0000          53.8438    ***
#             season_4      52.5240     3.0227      17.3767   0.0000          52.5240    ***
#         weathersit_4     -38.4895    55.4990      -0.6935   0.4880          38.4895
#                  hum     -32.9369     1.7156     -19.1979   0.0000          32.9369    ***
#                atemp      32.9222     7.6669       4.2940   0.0000          32.9222    ***
#                 bias      31.9610     3.0621      10.4376   0.0000          31.9610    ***
#         weathersit_3     -30.1026     4.0684      -7.3992   0.0000          30.1026    ***
#             season_2      28.5105     2.7826      10.2459   0.0000          28.5105    ***
#                 temp      22.2001     7.8432       2.8305   0.0047          22.2001     **
#         weathersit_2      -8.4138     2.1685      -3.8799   0.0001           8.4138    ***
#            windspeed      -3.2173     0.9896      -3.2510   0.0012           3.2173     **
#             season_3       1.3073     4.1527       0.3148   0.7529           1.3073

# --- 业务解读与交互分析 ---

# [主效应]
#   - temp: 显著正相关 (Coef: +22.20, P: 4.655e-03)
#     -> 解读: 温度越高，租赁量越多。
#   - hum: 显著负相关 (Coef: -32.94, P: 6.059e-81)
#   - weathersit_3: 显著负相关 (Coef: -30.10, P: 1.461e-13)
#     -> 解读: 天气为小雨/小雪时，租赁量显著减少。
#   - yr_1: 显著正相关 (Coef: +79.22, P: 5.641e-269)
#     -> 解读: 2012年(yr=1)的租赁量显著高于2011年(基准)。
#   - time_period_peak: 显著正相关 (Coef: +210.42, P: 0.000e+00)
#     -> 解读: 高峰时段(peak)的租赁量显著高于非高峰时段(low_hours)。
#   - holiday_1: 显著负相关 (Coef: -58.54, P: 4.117e-06)
#     -> 解读: 节假日(holiday=1)的租赁量更低 (通勤需求减少)。
#   - workingday_1: 显著负相关 (Coef: -53.84, P: 1.458e-17)

# [交互效应]
#   - temp_x_holiday: 显著正相关 (Coef: +77.89, P: 3.042e-03)
#     -> 解读: 节假日时，温度升高的正面效应被放大了 (休闲用户更喜欢好天气)。
#   - weathersit_x_peak: 显著负相关 (Coef: -58.55, P: 4.130e-19)
#     -> 解读: 高峰时段的坏天气(小雨/雪)导致租赁量额外大幅减少 (符合直觉，通勤者换乘)。
#   - hum_x_workingday: 显著正相关 (Coef: +86.47, P: 2.141e-19)
#     -> 解读: 湿度对工作日有额外的正向影响 (可能通勤者对湿度的容忍度高于非通勤者)。
# ======================================================================