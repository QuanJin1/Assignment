import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据并设置频率
df = pd.read_csv('goods_oil_wti.csv', parse_dates=['日期'], index_col='日期')
df = df.sort_index().asfreq('D')
series = df['值'].ffill()

# 2. 拟合 SARIMA 模型
#candidate_models = [
#    {'order': (1,0,1), 'seasonal_order': (1,1,1,12), 'desc': 'Best Model'},
#    {'order': (0,1,0), 'seasonal_order': (1,1,0,12), 'desc': 'try1'},
#    {'order': (1,1,1), 'seasonal_order': (0,1,1,6), 'desc': 'try2'},
#    {'order': (2,1,2), 'seasonal_order': (1,1,1,6),  'desc': 'try3'}
#]


order = (2, 1, 2)
seasonal_order = (1, 1, 1, 6)
model = SARIMAX(series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)
result = model.fit(disp=0)

# 3. 输出模型摘要
print(result.summary())

# 4. 使用数值索引进行预测
forecast_steps = 30
pred = result.get_prediction(start=len(series)-forecast_steps, dynamic=False)
pred_ci = pred.conf_int()

# 5. 绘制图表
plt.figure(figsize=(14, 7))
ax = series.plot(label='Observed', style='--')
pred.predicted_mean.plot(ax=ax, label='Forecast', lw=2)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='gray', alpha=0.2)
ax.set_title('SARIMA Model Forecasting Results', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend()
plt.tight_layout()

# 6. 保存图表避免交互式显示问题
plt.savefig('forecast.png', dpi=300)
plt.close()  # 关闭图形避免渲染警告