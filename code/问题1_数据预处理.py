"""
Author: duan
"""
# 一、数据预处理与长数据转换
import pandas as pd
import numpy as np

df = pd.read_csv('问题一数据.csv', encoding='utf-8')
# 1. 替换"OL"为NaN
# 首先识别所有Allele列
allele_cols = [col for col in df.columns if col.startswith('Allele')]
for col in allele_cols:
    df[col] = df[col].replace('OL', np.nan)

# 2. 应用固定阈值10 RFU（修改部分开始）
# 首先识别所有Height列
height_cols = [col for col in df.columns if col.startswith('Height')]

# 设置固定阈值为10
fixed_threshold = 10

# 创建一个掩码，标记要保留的数据点
mask = pd.DataFrame(False, index=df.index, columns=height_cols)
for i, row in df.iterrows():
    for h_col in height_cols:
        if pd.notna(row[h_col]) and row[h_col] >= fixed_threshold:
            mask.at[i, h_col] = True
# 修改部分结束

# 应用掩码 - 将不满足条件的Height和对应的Allele设为NaN
for i, h_col in enumerate(height_cols):
    a_col = f'Allele {i+1}'  # 对应的Allele列
    df.loc[~mask[h_col], h_col] = np.nan
    df.loc[~mask[h_col], a_col] = np.nan

# 3. 转换为长格式
# 首先准备要保留的列
id_cols = ['Sample File', '标签', 'Marker', 'Dye']
melt_cols = []

# 构建melt的变量名和值名
for i in range(1, 101):  # 假设最多有100个等位基因
    if f'Allele {i}' in df.columns and f'Height {i}' in df.columns:
        melt_cols.extend([f'Allele {i}', f'Height {i}'])
    else:
        break

# 执行melt操作
long_df = pd.melt(
    df,
    id_vars=id_cols,
    value_vars=melt_cols,
    var_name='Variable',
    value_name='Value'
)

# 分离Allele和Height信息
long_df['Type'] = long_df['Variable'].apply(lambda x: x.split()[0])
long_df['Index'] = long_df['Variable'].apply(lambda x: int(x.split()[1]))

# 转换回宽格式（但现在是长格式的结构）
long_df = long_df.pivot_table(
    index=id_cols + ['Index'],
    columns='Type',
    values='Value',
    aggfunc='first'  # 每个组合应该只有一个值
).reset_index()

# 清理数据
long_df = long_df.dropna(subset=['Allele', 'Height'])  # 删除Allele或Height为NaN的行
long_df['Height'] = pd.to_numeric(long_df['Height'], errors='coerce')  # 确保Height是数值

# 4. 保存为CSV
long_df.to_csv('数据1预处理.csv', index=False, encoding='utf-8-sig')

print("数据预处理完成，已保存为'数据1预处理.csv'")


# 二、删除多余不合理的的等位基因
import pandas as pd
# 1.读取Excel文件
df = pd.read_csv('数据1预处理.csv')

# 2. 对每个样本的每个Marker进行检查和修剪
def trim_alleles(group):
    label = group['标签'].iloc[0]  # 获取标签值（混合人数）
    max_allowed = label * 2  # 允许的最大等位基因数

    if len(group) > max_allowed:
        # 按Height升序排序，并保留前 max_allowed 个
        trimmed_group = group.sort_values('Height', ascending=False).head(max_allowed)
        return trimmed_group
    else:
        return group


# 按 Sample File 和 Marker 分组，并应用修剪函数
trimmed_df = df.groupby(['Sample File', 'Marker']).apply(trim_alleles).reset_index(drop=True)

# 3. 检查是否仍有超出限制的情况（可选）
check = trimmed_df.groupby(['Sample File', 'Marker', '标签']).size().reset_index(name='Count')
check['Max Allowed'] = check['标签'] * 2
check['Valid'] = check['Count'] <= check['Max Allowed']

print("修剪后检查：")
print(check[check['Valid'] == False])  # 如果仍有不合法情况，会打印出来

# 4. 保存修剪后的数据（直接覆盖原文件）
trimmed_df.to_csv('数据1预处理_修剪后.csv', index=False, encoding='utf-8-sig')
print("数据修剪完成，已保存为 '数据1预处理_修剪后.csv'")

