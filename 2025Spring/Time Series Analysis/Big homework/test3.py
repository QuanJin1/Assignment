import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 1. 数据加载与预处理
data = pd.read_csv('goods_oil_wti.csv')
data['日期'] = pd.to_datetime(data['日期'])
data.set_index('日期', inplace=True)
data.sort_index(inplace=True)  # 确保时间升序排列
ts = data['值']


# 2. ADF检验函数
def adf_test(timeseries, title=''):
    print(f'{"=" * 50}\nADF检验: {title}\n{"=" * 50}')
    result = adfuller(timeseries.dropna(), autolag='AIC', regression='ct')  # 包含常数和趋势项

    print(f'ADF统计量: {result[0]:.6f}')
    print(f'p值: {result[1]:.6f}')
    print(f'使用的滞后阶数 (lags): {result[2]}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.6f}')

    # 结果解读
    if result[1] < 0.05:
        print("-> 结论: 拒绝原假设，数据是平稳的")
    else:
        print("-> 结论: 不能拒绝原假设，数据是非平稳的")
    return result


# 3. 原始数据检验
orig_adf = adf_test(ts, '原始数据')

# 4. 一阶差分处理
ts_diff = ts.diff().dropna()

# 5. 差分数据检验
diff_adf = adf_test(ts_diff, '一阶差分数据')

# 6. 综合结果报告
print('\n\n' + '=' * 60)
print('平稳性检验综合报告'.center(60))
print('=' * 60)
print(f'原始数据ADF检验p值: {orig_adf[1]:.4f} → {"非平稳" if orig_adf[1] > 0.05 else "平稳"}')
print(f'一阶差分ADF检验p值: {diff_adf[1]:.4f} → {"非平稳" if diff_adf[1] > 0.05 else "平稳"}')
print('结论:')
if orig_adf[1] > 0.05 and diff_adf[1] < 0.05:
    print("原始序列是非平稳的，但一阶差分后变为平稳序列，符合单位根过程特征")
elif orig_adf[1] < 0.05:
    print("原始序列已经是平稳序列")
else:
    print("原始序列和一阶差分序列均为非平稳，可能需要更高阶差分或其他转换")