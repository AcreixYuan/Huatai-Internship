# -*- coding: utf-8 -*-
"""
Created on Mon May 22 2023

@author: ysy
"""
#%% --------------------------------- Import packages ---------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%% --------------------------------- Read data ---------------------------------
#读取excel
df_investment = pd.read_excel(r"Data/投资活动指数_原始数据.xlsx", index_col=0)

# 参数设置
panel_dates = pd.date_range('2018-12-30', '2023-05-14', freq='W') # 主成分分析区间

# 给所有指标取同差
df_investment = df_investment.diff(52)

# 重置样本区间为panel_dates
df_investment = df_investment.loc[panel_dates]

# 线性插值填充缺失值
df_investment = df_investment.interpolate(method='linear')

# zscore标准化所有指标
df_investment = df_investment.apply(lambda col: (col - col.mean()) / col.std())

#%% --------------------------------- PCA ---------------------------------
# 指明降到一维
pca = PCA(n_components = 1)

# 模型拟合, 改变方向
pca.fit(df_investment)

df_investment_reversed = df_investment.apply(lambda col: -col)

# 主成分分析结果
df_investment_pca = pd.DataFrame(pca.transform(df_investment_reversed), index=df_investment_reversed.index, columns=['Investment Index'])

# 保存主成分分析结果
df_investment_pca.to_excel(r"Result/投资活动指数_主成分分析结果.xlsx")

# %%
# 打印主成分分析权重
print(pca.components_)
# %%
df_investment_pca.plot()
# %%
