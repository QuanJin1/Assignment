import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Load and preprocess data
df = pd.read_csv('goods_oil_wti.csv', parse_dates=['日期'], index_col='日期')
df = df.sort_index()  # Ensure chronological order
series = df['值']  # Extract price series

# 2. Plot original time series
plt.figure(figsize=(14, 5))
plt.plot(series, label='WTI Crude Oil Price', color='blue')
plt.title('Original Time Series of WTI Crude Oil Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.savefig('01_original_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Plot ACF of original series
plt.figure(figsize=(14, 4))
plot_acf(series, lags=40, alpha=0.05)
plt.title('Autocorrelation (ACF) - Original Series')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.savefig('02_original_acf.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Compute first difference
diff_series = series.diff(periods=12).dropna()

# 5. Plot differenced series
plt.figure(figsize=(14, 5))
plt.plot(diff_series, label='First Difference', color='red')
plt.title('First Differenced Series of WTI Crude Oil Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Value')
plt.grid(True)
plt.legend()
plt.savefig('03_differenced_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Plot ACF and PACF of differenced series
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# ACF for differenced series
plot_acf(diff_series, lags=20, alpha=0.05, ax=axes[0], zero=False, auto_ylims=True)
axes[0].set_title('Autocorrelation (ACF) - First Differenced Series')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Correlation Coefficient')

# PACF for differenced series
plot_pacf(diff_series, lags=20, alpha=0.05, ax=axes[1], method='ywm', zero=False, auto_ylims=True)
axes[1].set_title('Partial Autocorrelation (PACF) - First Differenced Series')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Partial Correlation Coefficient')

plt.tight_layout()
plt.savefig('04_differenced_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()