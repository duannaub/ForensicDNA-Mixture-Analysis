"""
Author: duan
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（确保系统支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# #########################################一、数据预处理####################################################
# 1. 加载数据
df_noisy = pd.read_csv('附件1：不同人数的STR图谱数据.csv')  # 降噪前数据（含噪声）
df_clean = pd.read_csv('附件4：去噪后的STR图谱数据.csv')  # 降噪后数据（干净）

# 2. 数据清洗：处理缺失值和特殊值（如'OL'）
df_noisy.replace('OL', np.nan, inplace=True)
df_clean.replace('OL', np.nan, inplace=True)

# 3. 转换数据格式：宽表转长表
def melt_data(df):
    records = []
    for _, row in df.iterrows():
        sample = row['Sample File']
        marker = row['Marker']
        for i in range(1, 24):  # Allele 1 ~ Allele 23
            allele = row.get(f'Allele {i}')
            size = row.get(f'Size {i}')
            height = row.get(f'Height {i}')
            if pd.notna(allele) and pd.notna(height):
                records.append({
                    'Sample File': sample,
                    'Marker': marker,
                    'Allele': allele,
                    'Size': float(size) if pd.notna(size) else np.nan,
                    'Height': float(height) if pd.notna(height) else 0.0
                })
    return pd.DataFrame(records)

df_noisy_long = melt_data(df_noisy)
df_clean_long = melt_data(df_clean)

# 4. 对齐数据并生成标签
merged = pd.merge(
    df_noisy_long,
    df_clean_long,
    on=['Sample File', 'Marker', 'Allele', 'Size'],
    how='left',
    suffixes=('_noisy', '_clean')
)

# 标签规则：降噪后Height=0或不存在则为噪声（1）
merged['Label'] = np.where(
    (merged['Height_clean'].isna()) | (merged['Height_clean'] == 0),
    1,  # 噪声峰
    0   # 真实峰
)

# 5. 特征工程（'Height_ratio', 'Z_score', 'Distance', 'Height_noisy', 'Size'）
# 计算每个基因座的平均峰高和标准差
marker_stats = df_noisy_long.groupby('Marker')['Height'].agg(['mean', 'std']).reset_index()
marker_stats.columns = ['Marker', 'mean_height', 'std_height']
merged = pd.merge(merged, marker_stats, on='Marker', how='left')

# 特征1：峰高（噪声）与 该基因座平均峰高 的比值
merged['Height_ratio'] = (merged['Height_noisy'] / merged['mean_height']).astype(float)

# 特征2：峰高的Z-score（处理标准差为0的情况）
merged['Z_score'] = ((merged['Height_noisy'] - merged['mean_height']) /
                    merged['std_height'].replace(0, 1e-6)).astype(float)
merged['Z_score'].fillna(0, inplace=True)

# 特征3：与最近主峰的距离
def calc_min_distance(group):
    clean_sizes = group[group['Label'] == 0]['Size'].dropna().values
    if len(clean_sizes) == 0:
        group['Distance'] = -1.0  # 无主峰时设为-1（浮点型）
    else:
        group['Distance'] = group['Size'].apply(
            lambda x: float(np.min(np.abs(x - clean_sizes))))
    return group
merged = merged.groupby(['Sample File', 'Marker'], group_keys=False).apply(calc_min_distance)

# 6. 确保所有数值列为浮点型
numeric_cols = ['Height_noisy', 'Height_ratio', 'Z_score', 'Distance']
merged[numeric_cols] = merged[numeric_cols].astype(float)

# 7.检查Label=1（噪声峰）但Distance=0的异常记录
anomalies = merged[(merged['Label'] == 1) & (merged['Distance'] == 0)]
if not anomalies.empty:
    print("警告：存在噪声峰的Distance=0，请检查数据对齐！")
    print(anomalies[['Sample File', 'Marker', 'Allele', 'Size', 'Height_noisy', 'Height_clean']])

# 6. 保存预处理后的数据
merged.to_csv('preprocessed_data.csv', index=False, float_format='%.6f')  # 保留6位小数
print("预处理完成，数据已保存为 preprocessed_data.csv")

## #########################################二、可视化绘图1######################################################
# print(merged[['Height_ratio', 'Z_score', 'Distance']].describe())  # 查看数据分布特征

# 1.热力图（分析特征与标签之间有无线性相关性）
# 计算相关性矩阵
corr_matrix = merged[['Height_ratio', 'Z_score', 'Distance', 'Height_noisy', 'Size', 'Label']].corr()

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("特征与标签相关性热力图")
#plt.savefig("plot/特征相关性热力图.png", dpi=300, bbox_inches='tight')
plt.show()
# 结果：（1）五个特征与标签之间均没用明显的线性相关，Distance与Label的相关系数最大，为0.54，但也达不到最低相关要求（r=0.7）
#      （2）观察特征之间的相关系数，明显看到Z_score与Height_ratio（r=0.99）、Height_noisy与Z_score（r=0.0.88）、
#          Height_noisy与Height_ratio（r=0.86）这三对特征之间存在高度线性相关，其中Z_score与Height_ratio几乎完全相关，
#          是因为它们都是基于Height_noisy和基因座统计量（均值、标准差）计算的衍生特征。如果都拿来分析会出现多重共线性，
#          导致模型难以区分它们的独立贡献。于是我们可以先删除Z_score这个特征，先保留Height_noisy与Height_ratio这一对特征，进行下一步分析。


# 2.散点图矩阵（特征相关性分析）【能将0和1分开的就是好的分类特征】
# 选择需要分析的特征和标签、将Label转换为类别型（便于着色）
plot_data = merged[['Height_ratio', 'Distance', 'Height_noisy', 'Size', 'Label']]
plot_data['Label'] = plot_data['Label'].map({0: '真实峰', 1: '噪声峰'})

# 绘制散点图矩阵
sns.pairplot(plot_data, hue='Label', palette={'真实峰': 'green', '噪声峰': 'red'})
plt.suptitle("特征间关系分布图（按标签分类）", y=1.02)
#plt.savefig("plot/特征间关系分布图.png", dpi=300, bbox_inches='tight')
plt.show()
# 结果： （1）Size这个特征与所有特征几乎都不能很好地分离真实峰与噪声峰，首先可以将其删除；
#       （2）Distance与Height_ratio、Distance与Height_noisy这两队组合都能较好地将真实峰与噪声峰分开，因此保留Distance这个特征；
#       （3）观察Height_ratio与Height_noisy，明显可以看到这两个特征组合起来并没有分离真实峰与噪声峰，结合上述热力图的结果，应该考虑删除这二者中的一个。

# 3.箱线图（单特征与标签的关系）【真实峰的 Height_ratio 中位数应显著高于噪声峰】
plt.figure(figsize=(12, 4))

# Height_ratio
plt.subplot(1, 3, 1)
sns.boxplot(x='Label', y='Height_ratio', data=merged)
plt.title("峰高比值分布对比")

# Height_noisy
plt.subplot(1, 3, 2)
sns.boxplot(x='Label', y='Height_noisy', data=merged)
plt.title("峰高（降噪前）分布对比")

# Distance_to_nearest_peak
plt.subplot(1, 3, 3)
sns.boxplot(x='Label', y='Distance', data=merged)
plt.title("与最近主峰距离分布对比")

plt.tight_layout()
#plt.savefig("plot/特征分布对比图.png", dpi=300, bbox_inches='tight')
plt.show()

## #########################################三、XGBoost模型训练######################################################
data = pd.read_csv('preprocessed_data.csv')
# 1.选择特征和标签
features = ['Distance', 'Height_noisy']
X = data[features]
y = data['Label']

# 2.划分训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3.检查类别分布
print("训练集类别分布:\n", y_train.value_counts())
print("\n测试集类别分布:\n", y_test.value_counts())

# 4.初始化XGBoost分类器并训练模型
model = xgb.XGBClassifier(
    objective='binary:logistic',  # 二分类问题，且使用逻辑回归
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=10  # 正确位置：在构造函数中指定
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # 验证集
    verbose=True                  # 打印训练日志
)

## #########################################四、模型预测与基础评估###################################################
y_pred = model.predict(X_test)  # 预测测试集

# 输出评估报告
print("\n测试集评估报告:")
print(classification_report(y_test, y_pred))

# 特征重要性分析
importance = model.feature_importances_
feat_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(feat_importance)

## #########################################五、结果可视化####################################################
# 特征重要性柱状图
plt.figure(figsize=(8, 4))
plt.bar(feat_importance['Feature'], feat_importance['Importance'])
plt.title('XGBoost特征重要性')
plt.xlabel('特征')
plt.ylabel('重要性得分')
#plt.savefig('plot/特征重要性.png', dpi=300, bbox_inches='tight')
plt.close()


# 保存模型到文件
joblib.dump(model, 'xgboost_noise_classifier.pkl')
print("模型已保存为 xgboost_noise_classifier.pkl")

## #########################################六、进阶评估#####################################################
# 1.绘制单个混合样本降噪前后STR图谱对比（条形图）
def plot_str_comparison(sample_name, marker, df_noisy, df_clean):
    # 提取指定样本和基因座的数据
    noisy_peaks = df_noisy[(df_noisy['Sample File'] == sample_name) & (df_noisy['Marker'] == marker)]
    clean_peaks = df_clean[(df_clean['Sample File'] == sample_name) & (df_clean['Marker'] == marker)]

    # 绘制双条形图
    plt.figure(figsize=(12, 5))
    plt.bar(noisy_peaks['Size'], noisy_peaks['Height'], width=0.5, alpha=0.7, label='降噪前', color='red')
    plt.bar(clean_peaks['Size'], clean_peaks['Height'], width=0.3, alpha=0.7, label='降噪后', color='green')

    plt.title(f"STR图谱对比 - {sample_name} ({marker})")
    plt.xlabel("Size (bp)")
    plt.ylabel("Peak Height")
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig(f"plot/STR对比_{sample_name}_{marker}.png", dpi=300, bbox_inches='tight')
    plt.show()

# 调用
sample_name = df_noisy_long['Sample File'].iloc[0]  # 取第一行的样本名
plot_str_comparison(sample_name, 'D3S1358', df_noisy_long, df_clean_long)

# 2.计算等位基因保留率和噪声剔除率
def calculate_retention_elimination_rates(data, model, features):
    # 模型预测
    X = data[features]
    y_true = data['Label']
    y_pred = model.predict(X)

    # 计算等位基因保留率（真实峰中预测正确的比例）
    true_peaks = data[y_true == 0]
    retention_rate = accuracy_score(true_peaks['Label'], y_pred[y_true == 0])

    # 计算噪声剔除率（噪声峰中预测正确的比例）
    noise_peaks = data[y_true == 1]
    elimination_rate = accuracy_score(noise_peaks['Label'], y_pred[y_true == 1])

    print(f"等位基因保留率: {retention_rate:.4f}")
    print(f"噪声剔除率: {elimination_rate:.4f}")
    return retention_rate, elimination_rate

# 调用
features = ['Distance', 'Height_noisy']
data = pd.read_csv('preprocessed_data.csv')
model = joblib.load('xgboost_noise_classifier.pkl')
calculate_retention_elimination_rates(data, model, features)
# 结果：等位基因保留率: 1.0000
#      噪声剔除率: 1.0000

# 3.使用t-SNE降维可视化特征空间
def plot_tsne_features(data, features, label_column='Label'):
    # 提取特征和标签
    X = data[features]
    y = data[label_column]

    # t-SNE降维（2D）
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Label (0=真实峰, 1=噪声峰)')
    plt.title("t-SNE特征空间降维（噪声 vs 真实峰）")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    # plt.savefig("plot/tSNE_特征空间.png", dpi=300, bbox_inches='tight')
    plt.close()

# 调用示例
plot_tsne_features(data, features=['Height_ratio','Z_score', 'Distance'])


