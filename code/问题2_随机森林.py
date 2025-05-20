import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在所有matplotlib导入前设置
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from scipy.stats import entropy  # 如果用到熵计算
import os
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(file_path):
    """加载STR图谱数据"""
    df = pd.read_csv(file_path)
    print(f"数据加载成功，形状: {df.shape}")
    return df

def extract_mixing_ratio(df):
    """从样本文件名中提取混合比例信息"""
    pattern = r'RD14-0003-([0-9_]+)-([0-9;]+)-'

    # 初始化新列
    df['sample_id'] = range(len(df))
    df['contributor_count'] = np.nan
    df['contributor_ids'] = None
    df['mixing_ratio'] = None
    df['ratio_class'] = None

    for idx, row in df.iterrows():
        sample_file = row['Sample File']
        if pd.isna(sample_file):
            continue

        match = re.search(pattern, str(sample_file))
        if match:
            contributor_ids = match.group(1).split('_')
            mixing_ratios = match.group(2).split(';')

            # 贡献者数量
            contributor_count = len(contributor_ids)
            df.at[idx, 'contributor_count'] = contributor_count
            df.at[idx, 'contributor_ids'] = ','.join(contributor_ids)
            df.at[idx, 'mixing_ratio'] = ';'.join(mixing_ratios)

            # 创建比例类别标签
            df.at[idx, 'ratio_class'] = ':'.join(mixing_ratios)

    # 将contributor_count转换为整数
    df['contributor_count'] = df['contributor_count'].astype('Int64')

    return df

def analyze_mixing_ratios(df):
    """分析混合比例分布"""
    # 获取唯一的混合比例类别
    unique_ratios = df['ratio_class'].dropna().unique()
    print(f"发现 {len(unique_ratios)} 种不同的混合比例组合:")

    ratio_counts = df.groupby(['ratio_class', 'contributor_count']).size().reset_index(name='count')
    ratio_counts = ratio_counts.sort_values(['contributor_count', 'count'], ascending=[True, False])

    # 显示每种贡献者数量下的主要混合比例
    for count, group in ratio_counts.groupby('contributor_count'):
        print(f"\n{count}人混合样本的混合比例分布:")
        for _, row in group.iterrows():
            print(f"  {row['ratio_class']}: {row['count']}个样本")

    # 可视化混合比例分布
    plt.figure(figsize=(15, 10))

    # 按贡献者数量分组绘制
    for i, count in enumerate(sorted(ratio_counts['contributor_count'].unique())):
        if i >= 4:  # 最多只显示4个子图
            break
        plt.subplot(2, 2, i+1)
        sub_data = ratio_counts[ratio_counts['contributor_count'] == count]

        # 只显示前10个最常见的比例
        if len(sub_data) > 10:
            sub_data = sub_data.nlargest(10, 'count')

        # 绘制条形图
        ax = sns.barplot(x='ratio_class', y='count', data=sub_data, palette='viridis')
        plt.title(f'{count}人混合样本的混合比例分布', fontsize=14)
        plt.xlabel('混合比例', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数值标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 10),
                       textcoords='offset points')

    plt.tight_layout()
    plt.savefig('plots/混合比例分布分析.png', dpi=300, bbox_inches='tight')
    plt.close()

    return ratio_counts


def extract_robust_features(df):
    """提取稳健的特征，避免极端值和无穷大"""
    print("开始提取特征...")

    features_list = []

    for (sample_file, marker), group in df.groupby(['Sample File', 'Marker']):
        # 跳过缺失混合比例的样本
        if 'ratio_class' not in group.columns or group['ratio_class'].isnull().all():
            continue

        ratio_class = group['ratio_class'].iloc[0]
        contributor_count = group['contributor_count'].iloc[0]

        if pd.isna(ratio_class) or pd.isna(contributor_count):
            continue

        # 基本信息
        feature = {
            'sample_file': sample_file,
            'marker': marker,
            'ratio_class': ratio_class,
            'contributor_count': contributor_count,
            'sample_id': group['sample_id'].iloc[0]
        }

        # 提取等位基因和峰高数据
        alleles = []
        heights = []
        sizes = []
        non_ol_alleles = []

        for i in range(1, 21):  # 最多20个等位基因
            allele_col = f'Allele {i}'
            height_col = f'Height {i}'
            size_col = f'Size {i}'

            if (allele_col in group.columns and height_col in group.columns and
                    not group[allele_col].isnull().all() and not group[height_col].isnull().all()):

                allele = group[allele_col].iloc[0]
                height = group[height_col].iloc[0]

                if pd.notna(allele) and pd.notna(height) and height > 0:
                    alleles.append(str(allele))
                    heights.append(height)

                    # 记录非OL等位基因
                    if str(allele) != 'OL':
                        non_ol_alleles.append(str(allele))

                    # 获取size信息
                    if size_col in group.columns and not group[size_col].isnull().all():
                        size = group[size_col].iloc[0]
                        if pd.notna(size):
                            sizes.append(size)

        # === 基本等位基因特征 ===
        feature['allele_count'] = len(alleles)
        feature['non_ol_allele_count'] = len(non_ol_alleles)
        feature['ol_count'] = len(alleles) - len(non_ol_alleles)
        feature['ol_ratio'] = safe_divide(feature['ol_count'], len(alleles)) if len(alleles) > 0 else 0

        # === 峰高特征 - 确保没有极端值 ===
        if heights:
            # 使用稳健的统计方法，避免异常值的影响
            sorted_heights = sorted(heights, reverse=True)

            # 基本峰高统计
            feature['height_count'] = len(heights)
            feature['height_mean'] = np.mean(heights)
            feature['height_median'] = np.median(heights)
            feature['height_std'] = np.std(heights) if len(heights) > 1 else 0

            # 使用安全除法，避免除零错误
            feature['height_cv'] = safe_divide(feature['height_std'], feature['height_mean'])

            feature['height_max'] = max(heights)
            feature['height_min'] = min(heights)
            feature['height_range'] = feature['height_max'] - feature['height_min']
            feature['height_sum'] = sum(heights)

            # 保存前几个峰高及其比例
            max_peaks = min(10, len(sorted_heights))
            for i in range(max_peaks):
                feature[f'peak_{i + 1}_height'] = sorted_heights[i]
                feature[f'peak_{i + 1}_ratio'] = safe_divide(sorted_heights[i], feature['height_sum'])

                # 计算与第一个峰的比例（只计算前面的峰）
                if i > 0:
                    feature[f'peak_1_{i + 1}_ratio'] = safe_divide(sorted_heights[0], sorted_heights[i])

            # 相邻峰高比
            for i in range(len(sorted_heights) - 1):
                if i < max_peaks - 1:
                    feature[f'peak_{i + 1}_{i + 2}_ratio'] = safe_divide(sorted_heights[i], sorted_heights[i + 1])

            # 计算主要峰高占总峰高的比例
            for i in range(1, min(6, len(sorted_heights) + 1)):
                top_sum = sum(sorted_heights[:i])
                feature[f'top_{i}_sum_ratio'] = safe_divide(top_sum, feature['height_sum'])

            # === 特定比例特征 ===
            # 2人混合特征（恢复原始高精度版本）
            if contributor_count == 2 and len(sorted_heights) >= 2:
                ratio_values = ratio_class.split(':')

                # 计算理论峰高比例
                if len(ratio_values) == 2:
                    r1, r2 = int(ratio_values[0]), int(ratio_values[1])
                    ratio_max_min = max(r1, r2) / min(r1, r2) if min(r1, r2) > 0 else 0

                    # 观察到的峰高比例
                    observed_ratio = safe_divide(sorted_heights[0], sorted_heights[1])

                    # 计算与理论值的接近程度
                    if ratio_max_min > 0:
                        diff = abs(observed_ratio - ratio_max_min)
                        feature['ratio_theory_diff'] = diff
                        feature['ratio_theory_match'] = 1 / (1 + diff) if diff is not None else 0

                # 匹配常见的混合比例
                common_ratios = {
                    "1:1": 1,
                    "1:4": 4,
                    "1:9": 9
                }

                if len(sorted_heights) >= 2:
                    observed_ratio = safe_divide(sorted_heights[0], sorted_heights[1])

                    for ratio_name, ratio_value in common_ratios.items():
                        diff = abs(observed_ratio - ratio_value) if observed_ratio is not None else float('inf')
                        feature[f'ratio_match_{ratio_name}'] = 1 / (1 + diff) if diff is not None else 0

            # 3人混合新增特征
            if contributor_count == 3 and len(sorted_heights) >= 3:
                ratio_values = ratio_class.split(':')

                if len(ratio_values) == 3:
                    # 理论比例差异
                    sorted_ratios = sorted(map(int, ratio_values), reverse=True)
                    theory_ratio1 = safe_divide(sorted_ratios[0], sorted_ratios[1])
                    theory_ratio2 = safe_divide(sorted_ratios[0], sorted_ratios[2])
                    theory_ratio3 = safe_divide(sorted_ratios[1], sorted_ratios[2])

                    # 观测比例
                    obs_ratio1 = safe_divide(sorted_heights[0], sorted_heights[1])
                    obs_ratio2 = safe_divide(sorted_heights[0], sorted_heights[2])
                    obs_ratio3 = safe_divide(sorted_heights[1], sorted_heights[2])

                    # 差异特征
                    feature.update({
                        '3p_ratio_diff1': abs(theory_ratio1 - obs_ratio1),
                        '3p_ratio_diff2': abs(theory_ratio2 - obs_ratio2),
                        '3p_ratio_diff3': abs(theory_ratio3 - obs_ratio3)
                    })

                    # 动态比例模板匹配（需根据实际数据调整！）
                    common_3p_ratios = {
                        "1:4:1": [1, 4, 1],
                        "1:4:4": [1, 4, 4],  # 示例高频比例
                        "1:2:2": [1, 2, 2],
                        "1:9:1": [1, 9, 1],
                        "1:9:9": [1, 9, 9],
                        "1:1:1": [1, 1, 1]

                    }
                    for ratio_name, template in common_3p_ratios.items():
                        sorted_tmp = sorted(template, reverse=True)
                        tmp_ratios = [
                            safe_divide(sorted_tmp[0], sorted_tmp[1]),
                            safe_divide(sorted_tmp[0], sorted_tmp[2]),
                            safe_divide(sorted_tmp[1], sorted_tmp[2])
                        ]
                        diff = sum([
                            abs(tmp_ratios[0] - obs_ratio1),
                            abs(tmp_ratios[1] - obs_ratio2),
                            abs(tmp_ratios[2] - obs_ratio3)
                        ])
                        feature[f'3p_match_{ratio_name}'] = 1 / (1 + diff)

            # 4人混合特征
            if contributor_count == 4 and len(sorted_heights) >= 4:
                ratio_values = ratio_class.split(':')
                if len(ratio_values) == 4:
                    # 动态比例模板匹配
                    common_4p_ratios = {
                        "1:1:1:1": [1, 1, 1, 1],
                        "1:1:2:1": [1, 1, 2, 1],
                        "1:2:2:1": [1, 2, 2, 1],
                        "1:4:4:1": [1, 4, 4, 1],
                        "1:1:4:1": [1, 1, 4, 1],
                        "1:4:4:4": [1, 4, 4, 4],
                        "1:1:9:1": [1, 1, 9, 1],
                        "1:9:9:1": [1, 9, 9, 1]
                    }

                    # 计算观测比例
                    obs_ratios = [
                        safe_divide(sorted_heights[0], sorted_heights[1]),
                        safe_divide(sorted_heights[0], sorted_heights[2]),
                        safe_divide(sorted_heights[0], sorted_heights[3]),
                        safe_divide(sorted_heights[1], sorted_heights[2]),
                        safe_divide(sorted_heights[1], sorted_heights[3]),
                        safe_divide(sorted_heights[2], sorted_heights[3])
                    ]

                    for ratio_name, template in common_4p_ratios.items():
                        sorted_tmp = sorted(template, reverse=True)
                        tmp_ratios = [
                            safe_divide(sorted_tmp[0], sorted_tmp[1]),
                            safe_divide(sorted_tmp[0], sorted_tmp[2]),
                            safe_divide(sorted_tmp[0], sorted_tmp[3]),
                            safe_divide(sorted_tmp[1], sorted_tmp[2]),
                            safe_divide(sorted_tmp[1], sorted_tmp[3]),
                            safe_divide(sorted_tmp[2], sorted_tmp[3])
                        ]
                        diff = sum(abs(t - o) for t, o in zip(tmp_ratios, obs_ratios))
                        feature[f'4p_match_{ratio_name}'] = 1 / (1 + diff)

                    # 前四峰分布
                    top4_sum = sum(sorted_heights[:4])
                    for i in range(4):
                        feature[f'4p_top{i + 1}_ratio'] = safe_divide(sorted_heights[i], top4_sum)

                    # 相邻峰稳定性
                    ratio_diffs = []
                    for i in range(3):
                        ratio = safe_divide(sorted_heights[i], sorted_heights[i + 1])
                        ratio_diffs.append(abs(ratio - np.mean(ratio_diffs)) if ratio_diffs else 0)
                    feature['4p_ratio_std'] = np.std(ratio_diffs) if ratio_diffs else 0

                    # 衰减特征
                    decay_values = [
                        safe_divide(sorted_heights[1], sorted_heights[0]),
                        safe_divide(sorted_heights[2], sorted_heights[1]),
                        safe_divide(sorted_heights[3], sorted_heights[2])
                    ]
                    feature['4p_decay_mean'] = np.mean(decay_values)


            # 5人混合特征
            if contributor_count == 5 and len(sorted_heights) >= 5:
                ratio_values = ratio_class.split(':')
                if len(ratio_values) == 5:

                    # 动态比例模板匹配
                    common_5p_ratios = {
                        "1:1:1:1:1": [1, 1, 1, 1, 1],
                        "1:1:2:1:1": [1, 1, 2, 1, 1],
                        "1:1:2:9:1": [1, 1, 2, 9, 1],
                        "1:1:4:1:1": [1, 1, 4, 1, 1],
                        "1:4:4:4:1": [1, 4, 4, 4, 1],
                        "1:1:2:4:1": [1, 1, 2, 4, 1],
                        "1:9:9:9:1": [1, 9, 9, 9, 1]
                    }

                    # 计算观测比例（取前5峰的所有两两组合）
                    obs_ratios = []
                    for i, j in combinations(range(5), 2):
                        if i < j:  # 避免重复计算
                            obs_ratios.append(safe_divide(sorted_heights[i], sorted_heights[j]))

                    for ratio_name, template in common_5p_ratios.items():
                        sorted_tmp = sorted(template, reverse=True)
                        tmp_ratios = []
                        for i, j in combinations(range(5), 2):
                            if i < j:
                                tmp_ratios.append(safe_divide(sorted_tmp[i], sorted_tmp[j]))

                        diff = sum(abs(t - o) for t, o in zip(tmp_ratios, obs_ratios))
                        feature[f'5p_match_{ratio_name}'] = 1 / (1 + diff)

                    # 分布特征
                    quantiles = np.quantile(sorted_heights[:5], [0.2, 0.4, 0.6, 0.8])
                    feature.update({
                        '5p_q1_ratio': safe_divide(quantiles[0], sorted_heights[0]),
                        '5p_q3_ratio': safe_divide(quantiles[2], sorted_heights[0])
                    })

                    # 峰面积特征
                    top5_area = sum(sorted_heights[:5])
                    tail_area = sum(sorted_heights[5:]) if len(sorted_heights) > 5 else 0
                    feature['5p_area_ratio'] = safe_divide(top5_area, top5_area + tail_area)

                    # 稳定性特征
                    rank_diff = sum(abs(np.diff(sorted_heights[:5]))) / 4
                    feature['5p_rank_stability'] = 1 / (1 + rank_diff)

                    # 熵特征
                    normalized = np.array(sorted_heights[:5]) / sum(sorted_heights[:5])
                    feature['5p_entropy'] = -np.sum(normalized * np.log(normalized + 1e-9))


        # === 基因座特征 ===
        for known_marker in ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358',
                           'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433',
                           'vWA', 'TPOX', 'D18S51', 'AMEL', 'D5S818', 'FGA']:
            feature[f'marker_is_{known_marker}'] = 1 if marker == known_marker else 0

        features_list.append(feature)

    # 创建特征数据框
    if not features_list:
        print("警告: 未能生成有效特征")
        return None

    features_df = pd.DataFrame(features_list)

    # 处理缺失值和极端值
    features_df = clean_features(features_df)

    # 打印特征统计信息
    print(f"特征提取完成，共生成 {len(features_df)} 个样本，{len(features_df.columns)} 个特征")
    print(f"每个贡献者人数的样本数量分布:")
    print(features_df['contributor_count'].value_counts().sort_index())
    print(f"混合比例类别数量: {features_df['ratio_class'].nunique()}")

    return features_df



def safe_divide(a, b, default=0):
    """安全除法，避免除零错误"""
    if b is None or a is None:
        return default
    try:
        if b == 0:
            return default
        result = a / b
        # 检查结果是否为有限值
        if not np.isfinite(result):
            return default
        return result
    except:
        return default

def clean_features(df):
    """清理特征数据，处理缺失值和极端值"""
    # 复制数据框
    cleaned_df = df.copy()

    # 获取数值型列
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()

    # 从numeric_cols中移除不需要处理的列
    exclude_cols = ['sample_id', 'contributor_count']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # 用均值填充缺失值
    imputer = SimpleImputer(strategy='mean')
    cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])

    # 替换无穷大值
    for col in numeric_cols:
        # 将无穷大值替换为列的最大有限值或最小有限值
        max_val = cleaned_df[col][~np.isinf(cleaned_df[col])].max()
        min_val = cleaned_df[col][~np.isinf(cleaned_df[col])].min()

        # 替换正无穷为最大有限值的1.5倍
        cleaned_df.loc[cleaned_df[col] == np.inf, col] = max_val * 1.5 if not np.isnan(max_val) else 0

        # 替换负无穷为最小有限值的1.5倍
        cleaned_df.loc[cleaned_df[col] == -np.inf, col] = min_val * 1.5 if not np.isnan(min_val) else 0

        # 检查极端值（超过3倍标准差）
        mean = cleaned_df[col].mean()
        std = cleaned_df[col].std()

        if std > 0:
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std

            # 将超出界限的值截断
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)

    # 在clean_features函数中添加
    if 'contributor_count' in cleaned_df.columns:
        # 针对不同人数混合样本进行特征缩放
        for count in [3, 4, 5]:
            mask = cleaned_df['contributor_count'] == count
            cols = [c for c in cleaned_df.columns if c.startswith(f'{count}p')]
            if cols:
                scaler = RobustScaler()
                cleaned_df.loc[mask, cols] = scaler.fit_transform(cleaned_df.loc[mask, cols])
    return cleaned_df

def visualize_peak_height_ratios(features_df):
    """可视化峰高比例与混合比例的关系"""
    # 提取2人混合样本
    two_person_df = features_df[features_df['contributor_count'] == 2].copy()

    if len(two_person_df) > 0:
        # 提取主要的混合比例类别
        main_ratios = two_person_df['ratio_class'].value_counts().nlargest(5).index

        # 过滤数据
        filtered_df = two_person_df[two_person_df['ratio_class'].isin(main_ratios)]

        # 绘制峰高比例与混合比例的关系
        plt.figure(figsize=(15, 10))

        # 散点图：最高峰/次高峰比例
        plt.subplot(2, 2, 1)
        sns.boxplot(x='ratio_class', y='peak_1_2_ratio', data=filtered_df, palette='viridis')
        plt.title('最高峰/次高峰比例与混合比例的关系', fontsize=14)
        plt.xlabel('混合比例', fontsize=12)
        plt.ylabel('峰高比例 (最高/次高)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 散点图：最高峰占总峰高比例
        plt.subplot(2, 2, 2)
        sns.boxplot(x='ratio_class', y='peak_1_ratio', data=filtered_df, palette='viridis')
        plt.title('最高峰占总峰高比例与混合比例的关系', fontsize=14)
        plt.xlabel('混合比例', fontsize=12)
        plt.ylabel('最高峰/总峰高', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 散点图：前两个峰占总峰高比例
        plt.subplot(2, 2, 3)
        sns.boxplot(x='ratio_class', y='top_2_sum_ratio', data=filtered_df, palette='viridis')
        plt.title('前两个峰占总峰高比例与混合比例的关系', fontsize=14)
        plt.xlabel('混合比例', fontsize=12)
        plt.ylabel('前两个峰/总峰高', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 散点图：峰高变异系数
        plt.subplot(2, 2, 4)
        sns.boxplot(x='ratio_class', y='height_cv', data=filtered_df, palette='viridis')
        plt.title('峰高变异系数与混合比例的关系', fontsize=14)
        plt.xlabel('混合比例', fontsize=12)
        plt.ylabel('峰高变异系数', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('plots/峰高比例与混合比例关系.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_optimized_models(features_df):
    """训练针对不同贡献者数量优化的模型"""
    print("开始训练混合比例识别模型...")

    # 按贡献者数量分组处理
    results = {}

    for contributor_count, group in features_df.groupby('contributor_count'):
        print(f"\n训练 {contributor_count} 人混合样本的模型...")

        # 统计该组中的混合比例类别数量
        ratio_counts = group['ratio_class'].value_counts()
        print(f"发现 {len(ratio_counts)} 种不同的混合比例组合:")
        for ratio, count in ratio_counts.items():
            print(f"  {ratio}: {count}个样本")

        # 对混合比例类别进行编码
        le = LabelEncoder()
        group['ratio_class_encoded'] = le.fit_transform(group['ratio_class'])

        # 特征和目标
        feature_cols = [col for col in group.columns
                       if col not in ['sample_file', 'marker', 'ratio_class',
                                     'ratio_class_encoded', 'contributor_count', 'sample_id']]

        X = group[feature_cols]
        y = group['ratio_class_encoded']

        # 特征数据安全检查
        # 检查是否有无穷大或NaN值
        has_inf = np.any(np.isinf(X.values))
        has_nan = np.any(np.isnan(X.values))

        if has_inf or has_nan:
            print(f"警告: 发现无穷大或NaN值，进行额外清理...")
            # 替换无穷大为0
            X = X.replace([np.inf, -np.inf], 0)
            # 替换NaN为0
            X = X.fillna(0)

        # 打印特征范围
        print("\n特征值范围检查:")
        print(f"最小值: {X.min().min()}")
        print(f"最大值: {X.max().max()}")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 特征标准化 - 使用更稳健的方法
        scaler = StandardScaler()  # 标准化器
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"标准化过程出错: {e}")
            # 备用方案：使用更简单的方法
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
            print("使用原始未缩放特征...")

        # 为不同贡献者数量选择不同的模型
        if contributor_count == 2:
            # 2人混合样本 - 使用梯度提升
            print("\n训练2人混合样本优化模型...")
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                random_state=42
            )
        elif contributor_count == 3:
            # 3人混合样本 - 使用梯度提升
            print("\n训练3人混合样本优化模型...")
            model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=15,
                subsample=0.8,
                random_state=42
            )
        elif contributor_count == 4:
            # 4人混合样本 - 使用梯度提升
            print("\n训练4人混合样本优化模型...")
            model = GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=15,
                random_state=42
            )
        elif contributor_count == 5:
            # 5人混合样本 - 使用随机森林
            print("\n训练5人混合样本优化模型...")
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            # 其他贡献者数量的样本
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        # 训练模型
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"模型训练出错: {e}")
            print("尝试使用不同的模型...")
            # 备用模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)

        # 保存结果
        results[contributor_count] = {
            'model': model,
            'scaler': scaler,
            'label_encoder': le,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_cols': feature_cols
        }

        # 打印评估指标
        print(f"\n准确率: {accuracy:.4f}")

        # 计算类别加权的指标
        report = classification_report(y_test, y_pred, output_dict=True)
        if 'weighted avg' in report:
            print(f"加权精确率: {report['weighted avg']['precision']:.4f}")
            print(f"加权召回率: {report['weighted avg']['recall']:.4f}")
            print(f"加权F1分数: {report['weighted avg']['f1-score']:.4f}")

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            print(f"5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"交叉验证出错: {e}")

        # 绘制混淆矩阵
        cm = results[contributor_count]['confusion_matrix']

        # 如果类别太多，只显示主要的几个类别
        if len(le.classes_) > 10:
            # 找出测试集中出现次数最多的10个类别
            class_counts = np.bincount(y_test)
            top_indices = np.argsort(-class_counts)[:10]  # 取前10个最多的类别

            # 过滤混淆矩阵
            filtered_cm = np.zeros((10, 10), dtype=int)
            for i, true_idx in enumerate(top_indices):
                for j, pred_idx in enumerate(top_indices):
                    if true_idx < cm.shape[0] and pred_idx < cm.shape[1]:
                        filtered_cm[i, j] = cm[true_idx, pred_idx]

            cm = filtered_cm
            class_names = [le.inverse_transform([idx])[0] for idx in top_indices if idx < len(le.classes_)]
        else:
            class_names = le.classes_

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{contributor_count} 人混合样本的混淆矩阵', fontsize=14)
        plt.ylabel('真实混合比例', fontsize=12)
        plt.xlabel('预测混合比例', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/{contributor_count}人混合样本混淆矩阵.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 如果是随机森林或梯度提升，显示特征重要性
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            # 获取特征重要性
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            top_features = importance.head(15)
            sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
            plt.title(f'{contributor_count} 人混合样本的特征重要性', fontsize=14)
            plt.xlabel('重要性', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'plots/{contributor_count}人混合样本特征重要性.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("\n前15个重要特征:")
            print(top_features)

        # 绘制多类ROC曲线（仅当类别数量合理时）
        if len(le.classes_) <= 10:  # 避免类别太多导致图形过于复杂
            plt.figure(figsize=(10, 8))

            # 获取预测概率
            y_proba = model.predict_proba(X_test_scaled)

            # 为每个类别绘制ROC曲线
            for i in range(len(le.classes_)):
                fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                         label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'{contributor_count}人混合样本的多类ROC曲线', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'plots/{contributor_count}人混合样本ROC曲线.png', dpi=300, bbox_inches='tight')
            plt.close()

    return results

def reliable_voting_ensemble(features_df, models_dict):
    """使用可靠的投票集成方法进行预测"""
    print("\n使用可靠的投票集成方法进行样本级别预测...")

    # 按样本分组
    sample_predictions = defaultdict(list)

    # 对每个样本进行预测
    for _, row in features_df.iterrows():
        sample_file = row['sample_file']
        marker = row['marker']
        contributor_count = row['contributor_count']

        # 跳过没有对应模型的贡献者数量
        if contributor_count not in models_dict:
            continue

        true_ratio = row['ratio_class']

        # 获取模型信息
        model_info = models_dict[contributor_count]
        model = model_info['model']
        scaler = model_info['scaler']
        le = model_info['label_encoder']
        feature_cols = model_info['feature_cols']

        # 准备特征
        X_sample = pd.DataFrame([row[feature_cols]])

        # 缺失值处理
        X_sample = X_sample.fillna(0)

        # 无穷值处理
        X_sample = X_sample.replace([np.inf, -np.inf], 0)

        try:
            # 特征缩放
            X_sample_scaled = scaler.transform(X_sample)

            # 预测
            y_pred = model.predict(X_sample_scaled)[0]
            pred_ratio = le.inverse_transform([y_pred])[0]

            # 记录预测
            sample_predictions[sample_file].append({
                'marker': marker,
                'true_ratio': true_ratio,
                'pred_ratio': pred_ratio,
                'correct': true_ratio == pred_ratio
            })
        except Exception as e:
            print(f"预测错误 ({sample_file}, {marker}): {e}")

    # 使用多数投票聚合预测
    sample_accuracies = []

    for sample_file, predictions in sample_predictions.items():
        if not predictions:
            continue

        # 获取真实混合比例
        true_ratio = predictions[0]['true_ratio']

        # 候选预测计数
        pred_counts = {}
        for p in predictions:
            pred_ratio = p['pred_ratio']
            pred_counts[pred_ratio] = pred_counts.get(pred_ratio, 0) + 1

        # 多数投票
        final_pred = max(pred_counts.items(), key=lambda x: x[1])[0]

        # 计算标记级别的准确率
        marker_accuracy = sum(1 for p in predictions if p['correct']) / len(predictions)

        # 保存结果
        sample_accuracies.append({
            'sample_file': sample_file,
            'true_ratio': true_ratio,
            'pred_ratio': final_pred,
            'correct': true_ratio == final_pred,
            'marker_accuracy': marker_accuracy,
            'contributor_count': len(true_ratio.split(':'))
        })

    # 转换为DataFrame
    sample_accuracy_df = pd.DataFrame(sample_accuracies)

    # 计算总体准确率
    overall_accuracy = sample_accuracy_df['correct'].mean()
    print(f"样本级别总体准确率: {overall_accuracy:.4f}")

    # 按贡献者数量分组显示准确率
    for count, group in sample_accuracy_df.groupby('contributor_count'):
        print(f"{count} 人混合样本准确率: {group['correct'].mean():.4f} ({len(group)}个样本)")

    # 绘制样本级别准确率
    plt.figure(figsize=(12, 6))
    accuracy_by_count = sample_accuracy_df.groupby('contributor_count')['correct'].mean()

    ax = sns.barplot(x=accuracy_by_count.index, y=accuracy_by_count.values, palette='viridis')

    # 添加数值标签
    for i, v in enumerate(accuracy_by_count.values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center')

    plt.title('不同贡献者数量的混合比例识别准确率', fontsize=14)
    plt.xlabel('贡献者数量', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/不同贡献者数量准确率对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 显示一些常见错误
    if not sample_accuracy_df[~sample_accuracy_df['correct']].empty:
        print("\n常见错误模式:")
        error_df = sample_accuracy_df[~sample_accuracy_df['correct']].copy()
        error_patterns = error_df.groupby(['true_ratio', 'pred_ratio']).size().reset_index(name='count')
        error_patterns = error_patterns.sort_values('count', ascending=False).head(10)

        for _, row in error_patterns.iterrows():
            print(f"  真实比例 {row['true_ratio']} 被误判为 {row['pred_ratio']}: {row['count']}次")

    return sample_accuracy_df

def main():
    """主函数"""
    print("=" * 80)
    print("法医物证多人身份鉴定 - STR图谱混合比例识别分析（稳健版）".center(60))
    print("=" * 80)

    # 步骤1：加载数据
    file_path = "问题二数据.csv"

    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return

    print("\n第1步: 数据加载")
    print("-" * 50)
    df = load_data(file_path)

    # 步骤2：提取混合比例信息
    print("\n第2步: 提取混合比例信息")
    print("-" * 50)
    df = extract_mixing_ratio(df)

    # 步骤3：分析混合比例分布
    print("\n第3步: 分析混合比例分布")
    print("-" * 50)
    ratio_counts = analyze_mixing_ratios(df)

    # 步骤4：提取特征
    print("\n第4步: 提取稳健特征")
    print("-" * 50)
    features_df = extract_robust_features(df)

    if features_df is None:
        print("错误: 无法提取有效特征")
        return

    # 步骤5：可视化峰高比例与混合比例的关系
    print("\n第5步: 可视化峰高与混合比例关系")
    print("-" * 50)
    visualize_peak_height_ratios(features_df)

    # 步骤6：训练优化模型
    print("\n第6步: 训练优化模型")
    print("-" * 50)
    model_results = train_optimized_models(features_df)

    # 步骤7：使用可靠的投票集成方法进行预测
    print("\n第7步: 使用可靠的投票集成方法进行预测")
    print("-" * 50)
    sample_accuracy_df = reliable_voting_ensemble(features_df, model_results)

    print("\n分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()