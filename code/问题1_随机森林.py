import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # 必须在所有matplotlib导入前设置
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import joblib

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据预处理
def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    # 检查数据
    unique_samples = data['Sample File'].nunique()
    print(f"数据集中独立样本数量: {unique_samples}")
    print("\n各人数样本分布:")
    print(data.groupby('Sample File')['标签'].first().value_counts())

    return data


# 2. 特征工程（精简版）
def extract_features(data):
    features = []
    labels = []

    for sample_name, group in data.groupby('Sample File'):
        # 获取标签
        num_contributors = group['标签'].iloc[0]
        sample_features = {}

        # 基因座级别特征（精简为3个核心特征）
        for marker, marker_group in group.groupby('Marker'):
            heights = marker_group['Height']
            total_height = heights.sum()

            sample_features.update({
                f'{marker}_allele_counts': len(marker_group),                    # 该位点检测到的等位基因数量
                f'{marker}_height_ratio': heights.max() / (total_height + 1e-6), # 最高峰与总峰高的比值
                f'{marker}_height_entropy': -np.sum((heights / total_height) *   # 峰高的信息熵
                                                    np.log(heights / total_height + 1e-10))
            })

        # 全局特征（精简为2个核心特征）
        sample_features.update({
            'global_avg_allele': np.mean([v for k, v in sample_features.items()  # 所有位点allele_counts的平均值
                                          if '_allele_counts' in k]),
            'max_allele_in_sample': max([v for k, v in sample_features.items()   # 所有位点allele_counts的最大值
                                         if '_allele_counts' in k])
        })

        features.append(sample_features)
        labels.append(num_contributors)

    feature_df = pd.DataFrame(features)
    feature_df['人数'] = labels

    # 处理可能的缺失值
    feature_df = feature_df.fillna(0)

    return feature_df


# 3. 模型构建与评估（带特征选择）
def build_and_evaluate_model(feature_df):
    X = feature_df.drop('人数', axis=1)
    y = feature_df['人数']

    # 特征选择（基于随机森林重要性）
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="median"  # 选择重要性高于中位数的特征
    )
    X_reduced = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    print(f"\n特征数量: 原始{X.shape[1]} → 筛选后{X_reduced.shape[1]}")
    print("保留的特征:", selected_features.tolist())

    # 初始化随机森林模型
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 使用分层5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 获取交叉验证的预测结果
    y_pred = cross_val_predict(rf, X_reduced, y, cv=skf)

    # 评估模型
    print("\n=== 交叉验证结果 ===")
    print("整体准确率:", accuracy_score(y, y_pred))
    print("\n分类报告:\n", classification_report(y, y_pred))

    # 混淆矩阵可视化
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('混淆矩阵（交叉验证结果）')
    plt.xlabel('预测人数')
    plt.ylabel('实际人数')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # 添加保存混淆矩阵图片
    plt.close()

    # ROC曲线绘制（多分类）
    # 获取所有唯一类别
    classes = np.unique(y)
    n_classes = len(classes)  # 正确定义n_classes

    # 将标签二值化
    y_bin = label_binarize(y, classes=classes)

    # 获取预测概率（使用交叉验证预测概率）
    y_proba = cross_val_predict(rf, X_reduced, y, cv=skf, method='predict_proba')

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # 绘制所有ROC曲线
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0}人混合 (AUC = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))  # 使用classes[i]获取实际类别值

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('多类ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 打印AUC值
    print("\n各类别AUC值:")
    for i in range(n_classes):
        print(f"{classes[i]}人混合: {roc_auc[i]:.4f}")
    print(f"微平均AUC: {roc_auc['micro']:.4f}")

    # 特征重要性分析（使用筛选后的特征）
    rf.fit(X_reduced, y)
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n重要特征排序:")
    print(feature_importance)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')  # 添加保存特征重要性图片
    plt.close()

    # 保存特征列表到文件
    # 保存特征列名（在特征选择后）
    joblib.dump(selected_features.tolist(), 'feature_columns.pkl')  # 保存为pkl文件

    # 同时保存模型
    joblib.dump(rf, 'contributor_count_model.pkl')

    return rf, selected_features


# 主流程
def main():
    # 加载数据
    data = load_and_preprocess("数据1预处理_修剪后.csv")

    # 特征提取
    feature_df = extract_features(data)
    print("\n特征矩阵预览:")
    print(feature_df.head())

    # 直接使用模型和特征（不存储变量）
    build_and_evaluate_model(feature_df)

if __name__ == "__main__":
    main()
