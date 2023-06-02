# -*- coding: utf-8 -*-
"""
Created on Mon May 22 2023

@author: ysy
"""

#%% --------------------------------- Import packages ---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#%% --------------------------------- Read data ---------------------------------
#读取excel
df_CCFI = pd.read_excel(r"Data/CCFI_综合指数.xlsx", index_col=0)
df_container = pd.read_excel(r"Data/中国_全国主要港口_集装箱吞吐量_当月值.xlsx", index_col=0)
df_export = pd.read_excel(r"Data/中国_出口金额_当月同比.xlsx", index_col=0)

df_container_high_freq = pd.read_excel(r"Data/集装箱吞吐量_八大枢纽港口_当旬同比.xlsx", index_col=0)

# 参数设置
regression_dates = pd.date_range('2006-01-01', '2022-12-10', freq='M')
fitting_dates = pd.date_range('2021-05-01', '2022-12-10', freq='M') 

# 将样本转为月度，并取月度均值
df_CCFI = df_CCFI.resample('M').mean()
df_container = df_container.resample('M').last()
df_container_high_freq = df_container_high_freq.resample('M').mean()

# 将解释变量与被解释变量都转化为同比数据
df_CCFI = df_CCFI.diff(12)
df_container = df_container.diff(12)
df_export = df_export.diff(12)

# 将高频数据转化为增速数据
df_container_high_freq = df_container_high_freq.diff(12)

# 线性插值法填充缺失值
df_CCFI = df_CCFI.interpolate(method='linear')
df_container = df_container.interpolate(method='linear')
df_export = df_export.interpolate(method='linear')
df_container_high_freq = df_container_high_freq.interpolate(method='linear')

# 重置样本区间为regression_dates
df_CCFI = df_CCFI.loc[regression_dates]
df_container = df_container.loc[regression_dates]
df_export = df_export.loc[regression_dates]

df_container_high_freq = df_container_high_freq.loc[fitting_dates]

#%% --------------------------------- Linear Regression ---------------------------------
# 以df_export为被解释变量，df_CCFI和df_container为解释变量，用线性回归模型拟合
X = pd.concat([df_CCFI, df_container], axis=1)
X = sm.add_constant(X)
y = df_export

# 拟合线性回归模型
model = sm.OLS(y, X)
result = model.fit()

# 打印回归结果
print(result.summary())

#%% -------------------------------- Plot --------------------------------
# 月度数据拟合结果
y_fitted = result.fittedvalues
df_export_all = pd.concat([y, y_fitted], axis=1)

# 高频数据拟合结果
df_CCFI = df_CCFI.loc[fitting_dates]
new_X = pd.concat([df_CCFI, df_container_high_freq], axis=1)
new_X = sm.add_constant(new_X)
y_fitted_high_freq = result.predict(new_X)
df_export_high_freq = df_container_high_freq.copy()

# 画图
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_export_all.index, df_export_all.iloc[:, 0], label='Actual')
ax.plot(df_export_all.index, df_export_all.iloc[:, 1], label='Fitted')
ax.plot(df_export_high_freq.index, df_export_high_freq, label='High Frequency')
ax.legend()
ax.set_title('Export')
ax.set_xlabel('Date')
ax.set_ylabel('YoY')
plt.show()

#%% --------------------------------- Factorize ---------------------------------
# 重新读取数据以重新导出周度指标
df_CCFI = pd.read_excel(r"Data/CCFI_综合指数.xlsx", index_col=0)
df_container_high_freq = pd.read_excel(r"Data/集装箱吞吐量_八大枢纽港口_当旬同比.xlsx", index_col=0)

# 转为同比
df_CCFI = df_CCFI.diff(52)
df_container_high_freq = df_container_high_freq.diff(36)

# 将CCFI数据转为旬度
new_index = df_container_high_freq.index
df_CCFI_10_day = df_CCFI.reindex(new_index, method='ffill')

# 区间修正为fitting_dates
# df_CCFI_10_day = df_CCFI_10_day.loc[fitting_dates]
# df_container_high_freq = df_container_high_freq.loc[fitting_dates]

# 预测出口指标
factor_X = pd.concat([df_CCFI_10_day, df_container_high_freq], axis=1)
factor_X = sm.add_constant(factor_X)
factor_y = result.predict(factor_X)

factor_y.dropna(inplace=True)

# 将高频出口指标转换成出口强度指数（z-score）
factor_y = (factor_y - factor_y.mean())/factor_y.std()

# 画图
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(factor_y.index, factor_y)
ax.set_title('Export Factor')
plt.show()
# %%
print(df_CCFI_10_day)

# %%
print(factor_y)
# %%
