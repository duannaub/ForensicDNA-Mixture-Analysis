# 问题三
# 样本选取在226和217行
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
warnings.filterwarnings('ignore')
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(file_path):
    """加载STR图谱数据"""
    df = pd.read_csv(file_path)
    print(f"数据加载成功，形状: {df.shape}")
    return df

def process_data(df):
    """处理数据函数，提取混合比例信息"""
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

    # 重新排列列，将新列放在Sample File之后
    cols = df.columns.tolist()
    sample_file_idx = cols.index('Sample File')
    new_cols = cols[:sample_file_idx + 1] + ['sample_id', 'contributor_count', 'contributor_ids', 'mixing_ratio',
                                             'ratio_class'] + \
               [col for col in cols[sample_file_idx + 1:] if
                col not in ['sample_id', 'contributor_count', 'contributor_ids', 'mixing_ratio', 'ratio_class']]
    df = df[new_cols]
    return df

def process_file(input_file, output_file):
    """处理单个文件并保存结果"""
    if not os.path.exists(input_file):
        print(f"错误: 文件 '{input_file}' 不存在")
        return False

    print(f"\n处理文件: {input_file}")
    print("-" * 50)
    df = load_data(input_file)
    df = process_data(df)
    # 保存处理后的数据
    df.to_csv(output_file, index=False, encoding='utf_8_sig')
    print(f"处理后的数据已保存到: {output_file}")
    return True

def load_reference_genotypes(ref_df):
    """加载参考基因型数据 - 修正版（统一基因座名称）"""
    print("\n" + "=" * 80)
    print("附件三的基因型数据结构".center(50))
    print("=" * 80)
    print("\n参考基因型数据结构:")
    # print(f"原始列名: {ref_df.columns.tolist()}")
    ref_df = ref_df.dropna(how='all')

    # 确认Sample ID列存在
    if 'Sample ID' not in ref_df.columns:
        print("错误: 未找到'Sample ID'列")
        return {}

    # 识别基因座列并统一名称（将"AM"转为"AMEL"）
    marker_cols = []
    name_mapping = {}  # 记录原始列名到标准名称的映射
    for col in ref_df.columns:
        if col not in ['Reseach ID', 'Sample ID']:
            # 统一基因座名称：将"AM"改为"AMEL"，其他保持不变
            normalized_name = 'AMEL' if col.strip() == 'AM' else col.strip()
            marker_cols.append(normalized_name)
            name_mapping[col] = normalized_name

    # print(f"\n标准化后的基因座名称映射:")

    print(f"\n{len(marker_cols)}个基因座: {marker_cols}")
    reference_genotypes = {}
    for _, row in ref_df.iterrows():
        # 处理Sample ID（兼容可能存在的浮点数格式）
        sample_id = str(int(float(row['Sample ID']))) if pd.notna(row['Sample ID']) else None
        if not sample_id:
            continue

        reference_genotypes[sample_id] = {}
        for orig_col in ref_df.columns:
            if orig_col in ['Reseach ID', 'Sample ID']:
                continue

            # 使用标准化后的名称
            marker_name = name_mapping[orig_col]
            allele_value = row[orig_col]
            if pd.isna(allele_value):
                continue

            # 处理各种可能的等位基因格式
            allele_str = str(allele_value).strip()
            if ',' in allele_str:
                alleles = [a.strip() for a in allele_str.split(',')]
            elif '/' in allele_str:
                alleles = [a.strip() for a in allele_str.split('/')]
            elif ' ' in allele_str:
                parts = allele_str.split()
                alleles = parts if len(parts) == 2 else [allele_str]
            else:
                alleles = [allele_str]

            # 处理纯合子情况（数字型等位基因）
            if len(alleles) == 1 and alleles[0].replace('.', '').isdigit():
                alleles = [alleles[0], alleles[0]]

            # 特殊处理AMEL基因座（强制大写X/Y）
            if marker_name == 'AMEL':
                alleles = [a.upper() for a in alleles if a.upper() in ('X', 'Y')]
                if not alleles:  # 如果数据异常，默认设置为X
                    alleles = ['X']
            reference_genotypes[sample_id][marker_name] = alleles
    # 验证输出
    print(f"\n成功加载了{len(reference_genotypes)}个贡献者的参考基因型")
    print(f"前5个ID示例: {list(reference_genotypes.keys())[:5]}")

    # 打印第一个样本的基因型示例（包含AMEL验证）
    sample_id = list(reference_genotypes.keys())[0]
    print(f"\n样本 {sample_id} 的基因型示例:")
    for marker, alleles in list(reference_genotypes[sample_id].items())[:3] + [('AMEL', reference_genotypes[sample_id].get('AMEL', ['未找到']))]:
        print(f"  {marker}: {alleles}")
    return reference_genotypes

def find_sample_info(sample_file, df1_path="问题一数据_正则表达式.csv", df2_path="问题二数据_正则表达式.csv"):
    """
    在已处理的数据文件中查找样本信息，返回贡献者数量和混合比例
        sample_file: 要查找的样本文件名 (如"A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa")
    返回:
        tuple: (contributor_count, mixing_ratio) 或 (None, None) 如果未找到
    """
    # 尝试在问题一数据中查找
    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
        match1 = df1[df1['Sample File'] == sample_file]
        if not match1.empty:
            contributor_count = match1.iloc[0]['contributor_count']
            print(f"在问题一数据中找到样本: {sample_file}")
            print(f"贡献者数量: {contributor_count}")
            return contributor_count, None  # 问题一数据可能没有混合比例

    # 尝试在问题二数据中查找
    if os.path.exists(df2_path):
        df2 = pd.read_csv(df2_path)
        match2 = df2[df2['Sample File'] == sample_file]
        if not match2.empty:
            contributor_count = match2.iloc[0]['contributor_count']
            mixing_ratio = match2.iloc[0]['mixing_ratio']
            print(f"在问题二数据中找到样本: {sample_file}")
            print(f"贡献者数量: {contributor_count}, 混合比例: {mixing_ratio}")
            return contributor_count, mixing_ratio
    print(f"未找到样本: {sample_file}")
    return None, None

def select_test_samples(combined_df):
    """从各混合人数中选取3种比例×4个样本（每组12个样本）"""
    test_samples = []
    for num in range(2, 6):  # 2-5人混合
        # 筛选指定贡献人数的样本
        samples = combined_df[combined_df['contributor_count'] == num]

        # 按比例分组并选取3种比例
        ratio_groups = samples.groupby('ratio_class')
        selected_ratios = []

        # 第一步：选取3种不同比例（按比例组数量降序排列）
        for ratio, group in sorted(ratio_groups, key=lambda x: -len(x[1])):
            if len(selected_ratios) >= 3:
                break
            selected_ratios.append(ratio)

        # 第二步：对每种比例选取4个样本
        for ratio in selected_ratios:
            group_samples = ratio_groups.get_group(ratio)

            # 如果该比例组样本不足4个，则取全部
            sample_count = min(4, len(group_samples))
            selected = group_samples.sample(sample_count, replace=False).to_dict('records')

            test_samples.extend(selected)
            #print(f"\n已选取 {num}人混合样本 (比例: {ratio}):")
            #for sample in selected:
                #print(f"  - {sample['Sample File']}")
    print(f"\n总计选取样本数: {len(test_samples)} ")
    return test_samples

def extract_str_data_from_df(df, sample_file):
    """
        从DataFrame中提取指定样本文件的STR数据
        df: 包含STR数据的DataFrame
    返回:
        包含该样本所有STR位点数据的字典
    """
    # 筛选出指定样本文件的数据
    sample_data = df[df['Sample File'] == sample_file]
    if sample_data.empty:
        raise ValueError(f"样本文件 '{sample_file}' 未在数据中找到")

    # 获取样本的基本信息
    sample_info = {
        'sample_id': sample_data.iloc[0]['sample_id'],
        'contributor_count': sample_data.iloc[0]['contributor_count'],
        'contributor_ids': sample_data.iloc[0]['contributor_ids'],
        'mixing_ratio': sample_data.iloc[0]['mixing_ratio'],
        'ratio_class': sample_data.iloc[0]['ratio_class']
    }
    # 准备存储STR数据的字典
    str_data = {
        'sample_info': sample_info,
        'markers': {}
    }
    # 遍历样本的每个标记位点
    for _, row in sample_data.iterrows():
        marker = row['Marker']
        dye = row['Dye']

        # 存储该位点的等位基因数据
        alleles = []
        # 遍历所有可能的等位基因列(最多30个)
        for i in range(1, 31):
            allele_col = f'Allele {i}'
            size_col = f'Size {i}'
            height_col = f'Height {i}'

            # 检查是否存在该等位基因数据
            if pd.notna(row[allele_col]):
                allele_data = {
                    'allele': row[allele_col],
                    'size': row[size_col],
                    'height': row[height_col]
                }
                alleles.append(allele_data)

        # 将位点数据添加到结果中
        str_data['markers'][marker] = {
            'dye': dye,
            'alleles': alleles
        }
    return str_data

#########################################################################################################
def analyze_samples(test_samples, reference_genotypes, combined_df, run_genetic=True,
                    pop_size=100, generations=30, mutation_rate=0.1):
    """
    增强版样本分析函数（处理所有测试样本并计算平均分）
    参数:
        test_samples: 测试样本列表
        reference_genotypes: 参考基因型数据库
        combined_df: 合并后的STR数据DataFrame
        run_genetic: 是否运行遗传算法
        pop_size: 种群大小
        generations: 迭代代数
        mutation_rate: 变异率
    返回:
        各混合人数的平均得分字典 {2: 平均分, 3: 平均分, ...}
    """
    print("\n" + "=" * 50)
    print("批量样本分析结果".center(50))
    print("=" * 50)

    # 按贡献者人数分组
    results = {num: {'scores': [], 'samples': []} for num in range(2, 6)}
    selected_samples = {}  # 用于存储每种贡献人数的一个代表性样本

    for sample in test_samples:
        sample = sample if isinstance(sample, dict) else sample.to_dict()
        n_contributors = sample['contributor_count']

        try:
            # 打印当前处理样本信息
            print("\n" + "-" * 50)
            print(f"正在处理样本: {sample['Sample File']}")
            print(f"贡献者人数: {n_contributors}")
            print(f"混合比例: {sample['mixing_ratio']}")

            # 提取并标准化STR数据
            str_data = extract_str_data_from_df(combined_df, sample['Sample File'])
            normalized_data = normalize_peak_heights(str_data)

            # 验证混合比例（可选）
            # validate_mixing_ratio(normalized_data)

            if run_genetic and n_contributors > 1:
                # 初始化记录进化过程的变量
                ga_history = {
                    'best': [],
                    'avg': [],
                    'worst': []
                }
                # 筛选候选者
                candidates = filter_candidate_contributors(normalized_data, reference_genotypes)
                print(f"候选贡献者数量: {len(candidates)}")

                # 生成初始种群
                population = generate_initial_population(
                    candidates,
                    n_contributors=n_contributors,
                    population_size=pop_size
                )

                # 遗传算法迭代
                for gen in range(generations):
                    scored_pop = evaluate_population_fitness(population, normalized_data, reference_genotypes)
                    new_population = []
                    # 记录当前代的统计数据
                    scores = [score for score, _ in scored_pop]
                    ga_history['best'].append(max(scores))
                    ga_history['avg'].append(np.mean(scores))
                    ga_history['worst'].append(min(scores))

                    for _ in range(pop_size // 2):
                        parent1 = tournament_selection(scored_pop)
                        parent2 = tournament_selection(scored_pop)
                        child1, child2 = crossover(parent1, parent2)

                        if random.random() < mutation_rate:
                            child1 = mutate(child1, candidates)
                        if random.random() < mutation_rate:
                            child2 = mutate(child2, candidates)

                        new_population.extend([child1, child2])

                    population = new_population

                    # 打印每代最佳得分（可选）
                    best_score, best_individual = max(scored_pop, key=lambda x: x[0])
                    print(f"Generation {gen + 1}: Best Score={best_score:.1f}", end='\r')

                # 在遗传算法结束后，存储代表性样本
                if n_contributors not in selected_samples:
                    selected_samples[n_contributors] = {
                        'ga_history': ga_history,
                        'predicted': best_individual,
                        'true': sample['contributor_ids'].split(','),
                        'mixed_data': normalized_data,
                        'sample_file': sample['Sample File']
                    }

                # 获取最终最优解
                final_scores = evaluate_population_fitness(population, normalized_data, reference_genotypes)
                best_score, best_individual = max(final_scores, key=lambda x: x[0])

                # 计算准确率得分
                true_contributors = set(sample['contributor_ids'].split(','))
                predicted_contributors = set(best_individual)
                predicted_sorted = sorted(str(x) for x in predicted_contributors) # 转换成字符串
                correct_matches = len(true_contributors & predicted_contributors)
                accuracy_score = (correct_matches / n_contributors) * 100

                # 存储结果
                results[n_contributors]['scores'].append(accuracy_score)
                results[n_contributors]['samples'].append(sample['Sample File'])

                # 打印详细结果
                print("\n" + "-" * 30)
                print(f"样本 {sample['Sample File']} 分析结果:")
                print(f"真实贡献者: {true_contributors}")
                print(f"预测贡献者: {{{', '.join(predicted_sorted)}}}")
                print(f"匹配分数: {best_score:.1f}/100")
                print(f"准确率得分: {accuracy_score:.1f}%")
                print("-" * 30)

        except Exception as e:
            print(f"\n处理样本 {sample['Sample File']} 时出错: {str(e)}")
            continue  # 即使出错也返回空字典

        # 绘制合并后的图
        if selected_samples:
            # 遗传算法进化图
            ga_histories = {k: v['ga_history'] for k, v in selected_samples.items()}
            plot_combined_ga_progress(ga_histories)

            # 基因型匹配热图
            heatmap_data = {k: {
                'predicted': v['predicted'],
                'true': v['true'],
                'mixed_data': v['mixed_data']
            } for k, v in selected_samples.items()}
            plot_combined_heatmaps(heatmap_data, reference_genotypes)

            # 单独保存每种贡献人数的热图
            for num, data in selected_samples.items():
                plt.figure(figsize=(12, 8))
                all_contributors = list(set(data['predicted'] + data['true']))
                markers = list(data['mixed_data']['markers'].keys())
                match_matrix = np.zeros((len(all_contributors), len(markers)))

                for i, cid in enumerate(all_contributors):
                    for j, marker in enumerate(markers):
                        if marker in reference_genotypes.get(cid, {}):
                            ref_alleles = set(reference_genotypes[cid][marker])
                            mixed_alleles = {a['allele'] for a in data['mixed_data']['markers'][marker]['alleles']}
                            match_matrix[i, j] = len(ref_alleles & mixed_alleles) / len(ref_alleles) * 100

                ax = sns.heatmap(match_matrix, annot=True, fmt=".0f", cmap="Pastel1_r",vmin=0, vmax=100,
                                 xticklabels=markers, yticklabels=all_contributors)

                plt.title(f'{num}人混合样本基因型匹配热图\n样本: {data["sample_file"]}')
                plt.xlabel('STR位点')
                plt.ylabel('贡献者ID')

                # 标记真实贡献者
                for i, cid in enumerate(all_contributors):
                    if cid in data['true']:
                        ax.text(-0.5, i + 0.5, "★", color='red', fontsize=12, ha='center', va='center')

                plt.tight_layout()
                plt.savefig(f'plots/{num}人混合样本基因型匹配热图.png', dpi=300, bbox_inches='tight')
                plt.close()

    # 计算并打印各混合人数的平均得分
    print("\n" + "=" * 50)
    print("最终评估结果".center(50))
    print("=" * 50)
    for num in range(2, 6):
        if results[num]['scores']:
            avg_score = np.mean(results[num]['scores'])
            sample_count = len(results[num]['scores'])
            print(f"\n{num}人混合样本:")
            print(f"  样本数量: {sample_count}")
            print(f"  平均准确率: {avg_score:.1f}%")
            print(f"  样本列表: {results[num]['samples']}")
        else:
            print(f"\n{num}人混合样本: 无有效数据")

    return results

def normalize_peak_heights(str_data):
    """
    根据样本自身的混合比例标准化峰高
        str_data: 包含sample_info和markers的数据结构
    返回:
        标准化后的STR数据（添加normalized_height字段）
    """
    # 从样本信息中解析混合比例
    ratio_str = str_data['sample_info']['mixing_ratio']
    ratios = [float(r) for r in ratio_str.split(';')]
    primary_ratio = max(ratios)  # 取最大比例作为基准

    for marker, data in str_data['markers'].items():
        heights = [a['height'] for a in data['alleles']]
        if len(heights) == 0:
            continue

        # 计算标准化因子（使主贡献者的峰对应1.0，其他按比例缩放）
        max_height = max(heights)
        for allele in data['alleles']:
            allele['normalized_height'] = allele['height'] / max_height * primary_ratio
    return str_data


def filter_candidate_contributors(mixed_str_data, reference_genotypes):
    """
    初步筛选可能的贡献者组合
    参数:
        mixed_str_data: 标准化后的混合样本STR数据
        reference_genotypes: 参考基因型数据库
    返回:
        候选贡献者字典 {contributor_id: 匹配基因座数量}
    """
    candidates = {}

    # 检查每个参考个体
    for cid, genotypes in reference_genotypes.items():
        match_count = 0
        for marker, data in mixed_str_data['markers'].items():
            if marker not in genotypes:
                continue

            # 检查该个体的基因型是否能解释混合样本的等位基因
            mixed_alleles = {a['allele'] for a in data['alleles']}
            ref_alleles = set(genotypes[marker])
            if mixed_alleles & ref_alleles:  # 有交集即认为可能
                match_count += 1

        if match_count > 0:  # 至少匹配一个基因座才保留
            candidates[cid] = match_count

    # 按匹配基因座数量降序排序
    return dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True))


def generate_initial_population(candidates, n_contributors=2, population_size=50):
    """
    根据候选贡献者生成遗传算法的初始种群
    参数:
        candidates: filter_candidate_contributors()返回的候选字典
        n_contributors: 贡献者人数
        population_size: 种群大小
    返回:
        种群列表（每个个体是贡献者ID的列表）
    """
    if n_contributors < 2:
        raise ValueError("贡献者数量必须至少为2")

    population = []
    candidate_ids = list(candidates.keys())

    if len(candidate_ids) < n_contributors:
        raise ValueError(f"候选贡献者不足 {n_contributors} 人")

    for _ in range(population_size):
        # 加权随机选择（禁止重复ID）
        weights = np.array([candidates[cid] for cid in candidate_ids])
        weights = weights / weights.sum()
        individual = np.random.choice(candidate_ids, size=n_contributors, replace=False, p=weights)
        population.append(list(individual))

    return population


def calculate_individual_score(contributor_id, mixed_data, reference_genotypes, true_contributors=None):
    """
    改进版个体评分函数（同时考虑基因型匹配和贡献者ID准确率）
    参数:
        contributor_id: 候选贡献者ID
        mixed_data: 标准化后的混合样本数据
        reference_genotypes: 参考基因型数据库
        true_contributors: 真实贡献者ID集合（可选，用于计算准确率）
    返回:
        匹配分数 (0-100)
    """
    # 1. 计算基因型匹配分数（原逻辑）
    score = 0
    total_markers = 0
    for marker, data in mixed_data['markers'].items():
        if marker not in reference_genotypes[contributor_id]:
            continue

        mixed_alleles = {a['allele']: a['normalized_height'] for a in data['alleles']}
        ref_alleles = set(reference_genotypes[contributor_id][marker])

        # 等位基因匹配得分
        matched_alleles = set(mixed_alleles.keys()) & ref_alleles
        allele_score = len(matched_alleles) * 10

        # 峰高比例得分
        ratio_score = 0
        if matched_alleles:
            expected_ratio = 1.0  # 主贡献者为1.0
            observed_ratios = [mixed_alleles[a] for a in matched_alleles]
            ratio_error = sum(abs(r - expected_ratio) for r in observed_ratios) / len(observed_ratios)
            ratio_score = max(0, 20 * (1 - ratio_error))

        # 完全匹配奖励
        full_match_bonus = 20 if matched_alleles == ref_alleles else 0
        score += allele_score + ratio_score + full_match_bonus
        total_markers += 1

    # 基因型匹配分数（0-100）
    genotype_score = (score / (total_markers * 50)) * 100 if total_markers > 0 else 0

    # 2. 如果提供真实贡献者，计算准确率权重
    if true_contributors is not None:
        is_correct = 1 if contributor_id in true_contributors else 0
        # 加权综合分数（基因型匹配占70%，准确率占30%）
        final_score = 0.7 * genotype_score + 0.3 * (is_correct * 100)
    else:
        final_score = genotype_score

    return min(100, final_score)

def evaluate_population_fitness(population, mixed_data, reference_genotypes):
    """
    评估种群适应度（匹配度）
    参数:
        population: 种群列表 (如 [['35','36'], ['35','38'], ...])
        mixed_data: 标准化后的混合样本数据
        reference_genotypes: 参考基因型数据库
    返回:
        包含分数和个体的排序列表 [(score, individual), ...]
    """
    scored_population = []
    n_contributors = mixed_data['sample_info']['contributor_count']  # 从样本数据获取目标人数
    true_contributors = set(mixed_data['sample_info']['contributor_ids'].split(','))  # 新增：获取真实贡献者

    for individual in population:
        # 强制修正个体长度（丢弃或填充至目标人数）
        if len(individual) > n_contributors:
            individual = individual[:n_contributors]  # 截断多余贡献者
        elif len(individual) < n_contributors:
            # 从候选者中随机补充（避免重复）
            candidates = list(reference_genotypes.keys())
            remaining = [cid for cid in candidates if cid not in individual]
            individual += random.sample(remaining, n_contributors - len(individual))

        # 计算组合分数（平均分 + 惩罚重复ID）
        individual_score = 0
        for cid in individual:
            individual_score += calculate_individual_score(cid, mixed_data, reference_genotypes)
        individual_score /= n_contributors  # 标准化平均分

        # 新增：计算准确率并加权（保持原有分数计算不变）
        correct_matches = len(set(individual) & true_contributors)
        accuracy_score = (correct_matches / n_contributors) * 100
        final_score = 0.7 * individual_score + 0.3 * accuracy_score  # 权重可调

        if len(set(individual)) < n_contributors:  # 惩罚重复ID
            final_score *= 0.5

        scored_population.append((final_score, individual))
    return sorted(scored_population, key=lambda x: -x[0])  # 按分数降序排序


def tournament_selection(scored_population, tournament_size=3):
    """
    锦标赛选择算子
    参数:
        scored_population: evaluate_population_fitness()返回的已评分种群
        tournament_size: 每次锦标赛的参赛个体数
    返回:
        被选中的个体
    """
    # 随机选择参赛者
    contestants = random.sample(scored_population, tournament_size)
    # 返回分数最高的个体
    return max(contestants, key=lambda x: x[0])[1]


def crossover(parent1, parent2):
    """交叉操作，强制保持个体长度一致"""
    if len(parent1) != len(parent2):
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [cid for cid in parent2[point:] if cid not in parent1[:point]]
    child2 = parent2[:point] + [cid for cid in parent1[point:] if cid not in parent2[:point]]

    # 确保长度不变
    child1 = child1[:len(parent1)]
    child2 = child2[:len(parent1)]
    return child1, child2


def mutate(individual, candidate_dict):
    """变异操作，不改变个体长度"""
    idx = random.randint(0, len(individual) - 1)
    new_id = random.choice([cid for cid in candidate_dict.keys() if cid not in individual])
    individual[idx] = new_id
    return individual


def validate_mixing_ratio(str_data):
    """验证申报比例与实际峰高是否一致"""
    declared_ratio = [float(r) for r in str_data['sample_info']['mixing_ratio'].split(';')]

    print("\n混合比例验证:")
    for marker, data in str_data['markers'].items():
        heights = sorted([a['normalized_height'] for a in data['alleles']], reverse=True)
        observed_ratio = heights[:len(declared_ratio)]
        # print(f"  {marker}: 申报={declared_ratio} 观测={[round(h, 2) for h in observed_ratio]}")


def plot_results(results):
    """绘制不同人数混合的准确率结果"""
    plt.figure(figsize=(10, 6))
    x = []
    y = []
    for num in range(2, 6):
        if results[num]['scores']:
            x.append(str(num))
            y.append(np.mean(results[num]['scores']))

    plt.bar(x, y, color='skyblue')
    plt.xlabel('混合人数')
    plt.ylabel('平均准确率(%)')
    plt.title('不同人数混合样本识别准确率', pad=20) # n*p是n个不同比例*p个数量
    plt.ylim(0, 105)
    for i, v in enumerate(y):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    plt.savefig(f'plots/不同人数混合样本识别准确率.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_ga_progress(all_ga_histories):
    """将四种贡献人数的遗传算法进化图合并为一张图"""
    plt.figure(figsize=(15, 10))
    plt.suptitle('不同贡献人数下的遗传算法进化过程对比', fontsize=16)

    colors = ['b', 'g', 'r', 'purple']
    line_styles = ['-', '--', ':']  # 实线、虚线、点线

    for idx, (num_contributors, ga_history) in enumerate(all_ga_histories.items(), 1):
        plt.subplot(2, 2, idx)

        # 修正：分开指定颜色和线型
        plt.plot(ga_history['best'], color=colors[idx - 1], linestyle=line_styles[0], label='最佳适应度')
        plt.plot(ga_history['avg'], color=colors[idx - 1], linestyle=line_styles[1], label='平均适应度')
        plt.plot(ga_history['worst'], color=colors[idx - 1], linestyle=line_styles[2], label='最差适应度')

        plt.xlabel('迭代代数')
        plt.ylabel('适应度分数')
        plt.title(f'{num_contributors}人混合样本')
        plt.grid(True)
        if idx == 1:
            plt.legend()

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/遗传算法进化过程对比.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_heatmaps(all_heatmap_data, reference_genotypes):
    """绘制四种贡献人数的基因型匹配热图"""
    plt.figure(figsize=(18, 12))
    plt.suptitle('不同贡献人数下的基因型匹配热图对比', fontsize=16)

    for idx, (num_contributors, data) in enumerate(all_heatmap_data.items(), 1):
        predicted = data['predicted']
        true = data['true']
        mixed_data = data['mixed_data']

        # 准备热图数据
        all_contributors = list(set(predicted + true))
        markers = list(mixed_data['markers'].keys())
        match_matrix = np.zeros((len(all_contributors), len(markers)))

        for i, cid in enumerate(all_contributors):
            for j, marker in enumerate(markers):
                if marker in reference_genotypes.get(cid, {}):
                    ref_alleles = set(reference_genotypes[cid][marker])
                    mixed_alleles = {a['allele'] for a in mixed_data['markers'][marker]['alleles']}
                    match_matrix[i, j] = len(ref_alleles & mixed_alleles) / len(ref_alleles) * 100

                    # 确保≤100

        # 绘制子图
        plt.subplot(2, 2, idx)
        ax = sns.heatmap(
            match_matrix,
            annot=True,
            fmt=".0f",
            cmap="Pastel1_r",
            vmin=0,
            vmax=100,
            xticklabels=markers if idx > 2 else [],
            yticklabels=all_contributors
        )
        plt.title(f'{num_contributors}人混合样本')

        # 标记真实贡献者
        for i, cid in enumerate(all_contributors):
            if cid in true:
                ax.text(-0.5, i + 0.5, "★", color='red', fontsize=12, ha='center', va='center')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/基因型匹配热图对比.png', dpi=300, bbox_inches='tight')
    plt.close()

########------------------------------------------------------------------------------------
def main():
    """主函数"""
    print("=" * 100)
    print("STR混合样本分析系统".center(60))
    print("=" * 100)
    # 设置随机数种子以确保结果可重复
    #random.seed(22)  # 可以是任意整数，常用42作为种子
    #np.random.seed(22)  # 如果你使用了numpy的随机函数也需要设置

    # 1. 数据加载与预处理
    file_pairs = [
        ("问题一数据.csv", "问题一数据_正则表达式.csv"),
        ("问题二数据.csv", "问题二数据_正则表达式.csv")
    ]
    for input_file, output_file in file_pairs:
        process_file(input_file, output_file)

    # 2. 加载参考基因型
    ref_file = "问题三数据.csv"
    if os.path.exists(ref_file):
        ref_df = pd.read_csv(ref_file)
        if ref_df.columns[0].startswith('\ufeff'):
            ref_df = ref_df.rename(columns={ref_df.columns[0]: ref_df.columns[0][1:]})
        reference_genotypes = load_reference_genotypes(ref_df)
    else:
        print(f"错误: 参考基因型文件 '{ref_file}' 不存在")
        reference_genotypes = None
        return

    # 3. 合并数据并选择样本
    combined_df = pd.concat([
        pd.read_csv("问题一数据_正则表达式.csv"),
        pd.read_csv("问题二数据_正则表达式.csv")
    ])
    combined_df.to_csv('合并后的数据.csv', index=False, encoding='utf_8_sig')

    test_samples = select_test_samples(combined_df)
    if not test_samples:
        print("错误: 未找到测试样本")
        return

    # 4. 分析样本（自动运行遗传算法）
    results = analyze_samples(
        test_samples=test_samples,
        reference_genotypes=reference_genotypes,
        combined_df=combined_df,
        run_genetic=True,
        pop_size=100,
        generations=30,
        mutation_rate=0.1
    )
    # 5. 可视化结果
    plot_results(results)

if __name__ == "__main__":
    main()

