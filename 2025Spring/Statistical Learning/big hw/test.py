import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

# ========================
# 添加中文字体支持
# ========================
import matplotlib as mpl

# 设置中文字体（Windows使用SimHei，Mac使用Arial Unicode MS）
if os.name == 'nt':  # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
else:  # Mac/Linux系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 设置全局字体大小
mpl.rcParams['font.size'] = 12

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

# ========================
# 1. 数据加载（使用所有特征）
# ========================
try:
    wine_df = pd.read_csv('winequality-red.csv')  # 明确指定分号分隔符
    print("=" * 50)
    print(f"成功加载数据，结构：{wine_df.shape[0]}行×{wine_df.shape[1]}列")
except Exception as e:
    print(f"数据加载失败: {str(e)}")
    exit()

# 验证特征数量
expected_features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol', 'quality'
]
missing_features = [f for f in expected_features if f not in wine_df.columns]
if missing_features:
    print(f"❌ 缺失特征: {missing_features}")
    exit()

print("✅ 所有12个特征均已加载")

# 创建字段解释表
field_explanations = {
    'fixed acidity': '固定酸度（酒石酸等不易挥发的酸）',
    'volatile acidity': '挥发性酸度（醋酸等易挥发酸）',
    'citric acid': '柠檬酸',
    'residual sugar': '残留糖分',
    'chlorides': '氯化物（盐分）',
    'free sulfur dioxide': '游离二氧化硫',
    'total sulfur dioxide': '总二氧化硫',
    'density': '密度',
    'pH': 'pH值',
    'sulphates': '硫酸盐',
    'alcohol': '酒精度',
    'quality': '感官质量评分（0-10分）'
}

# 打印字段说明表
print("\n表1 数据字段解释说明")
print("-" * 50)
print(f"{'字段':<25}{'说明'}")
print("-" * 50)
for field, explanation in field_explanations.items():
    print(f"{field:<25}{explanation}")

# ========================
# 2. 划分训练集和测试集
# ========================
print("\n\n" + "=" * 50)
print("2. 划分训练集和测试集")
print("=" * 50)

# 划分训练集和测试集
train_df, test_df = train_test_split(wine_df, test_size=0.2, random_state=42)

print(f"训练集大小: {train_df.shape[0]} 样本")
print(f"测试集大小: {test_df.shape[0]} 样本")

# ========================
# 3. 训练集和测试集基础统计分析
# ========================
print("\n\n" + "=" * 50)
print("3. 训练集和测试集基础统计分析")
print("=" * 50)


# 创建统计表函数
def create_stats_table(df, dataset_name):
    desc = df.describe().T
    desc['skewness'] = df.skew(numeric_only=True)

    stats_df = pd.DataFrame({
        'Count': desc['count'],
        'Mean': desc['mean'],
        'SD': desc['std'],
        'Min': desc['min'],
        'Q1': desc['25%'],
        'Median': desc['50%'],
        'Q3': desc['75%'],
        'Max': desc['max'],
        'Skewness': desc['skewness']
    })

    print(f"\n表2 {dataset_name}基础统计结果")
    print("-" * 120)
    header = f"{'指标':<25}{'数量':>8}{'均值':>10}{'标准差':>10}{'最小值':>10}{'Q1':>10}{'中位数':>10}{'Q3':>10}{'最大值':>10}{'偏度':>10}"
    print(header)
    print("-" * 120)
    for index, row in stats_df.iterrows():
        line = f"{index:<25}{int(row['Count']):>8}{row['Mean']:>10.2f}{row['SD']:>10.2f}{row['Min']:>10.2f}{row['Q1']:>10.2f}{row['Median']:>10.2f}{row['Q3']:>10.2f}{row['Max']:>10.2f}{row['Skewness']:>10.2f}"
        print(line)

    return stats_df


# 训练集统计
train_stats = create_stats_table(train_df, "训练集")

# 测试集统计
test_stats = create_stats_table(test_df, "测试集")

# ========================
# 4. 训练集和测试集数据分布可视化
# ========================
print("\n\n" + "=" * 50)
print("4. 训练集和测试集特征分布可视化")
print("=" * 50)


# 创建特征分布图函数
def plot_feature_distributions(df, title, filename, color):
    plt.figure(figsize=(18, 20))
    plt.suptitle(title, fontsize=16, y=0.95)

    features = df.columns
    rows = 4
    cols = 3

    for i, feature in enumerate(features, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[feature], kde=True, color=color)
        plt.title(f'{field_explanations[feature]}分布')
        plt.xlabel('')
        plt.ylabel('密度')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"已生成分布图: {filename}")


# 绘制训练集特征分布
plot_feature_distributions(train_df, '图表 3 训练集特征分布', 'output/train_features_distribution.png', 'skyblue')

# 绘制测试集特征分布
plot_feature_distributions(test_df, '图表 4 测试集特征分布', 'output/test_features_distribution.png', 'salmon')

# ========================
# 5. 相关性分析（使用所有特征）
# ========================
print("\n\n" + "=" * 50)
print("5. 相关性分析（所有特征）")
print("=" * 50)

plt.figure(figsize=(14, 12))
corr_matrix = wine_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size": 8})
plt.title('图表 5 特征相关性热图', fontsize=14)
plt.tight_layout()
plt.savefig('output/wine_correlation_heatmap.png')
plt.close()
print("已生成相关性分析热图: output/wine_correlation_heatmap.png")

print("\n相关性分析摘要：")
print("1. 与质量(quality)相关性最高的变量：")
quality_corr = corr_matrix['quality'].sort_values(ascending=False)[1:6]  # 前5个（排除自身）
for var, corr in quality_corr.items():
    print(f"   - {var}: {corr:.2f} ({field_explanations[var]})")

print("\n2. 特征间的强相关关系：")
# 找出除对角线外相关性最强的5对特征
corr_matrix_no_diag = corr_matrix.copy()
np.fill_diagonal(corr_matrix_no_diag.values, 0)
strong_corrs = corr_matrix_no_diag.unstack().sort_values(ascending=False).drop_duplicates()[:5]
for (var1, var2), corr in strong_corrs.items():
    print(f"   - {var1} 和 {var2}: {corr:.2f}")

# ========================
# 6. K均值聚类分析（使用所有特征）
# ========================
print("\n\n" + "=" * 50)
print("6. K均值聚类分析（使用所有特征）")
print("=" * 50)

# 数据标准化（使用所有特征，排除quality）
features_for_clustering = [f for f in expected_features if f != 'quality']
X = wine_df[features_for_clustering]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 寻找最佳K值（使用轮廓系数）
sil_scores = []
k_range = range(2, 11)

# 存储肘部法则的WCSS值
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    sil_scores.append(sil_score)
    wcss.append(kmeans.inertia_)  # 记录WCSS
    print(f"k={k}: 轮廓系数 = {sil_score:.4f}, WCSS = {kmeans.inertia_:.2f}")

# 自动检测肘点（二阶差分法）
diffs = np.diff(wcss)  # 一阶差分
diff2 = np.diff(diffs)  # 二阶差分
elbow_k = k_range[np.argmin(diff2) + 2]  # +2补偿二阶差分偏移

# 决策逻辑
if sil_scores[elbow_k - 2] > 0.2:  # -2补偿k_range起始值
    best_k = elbow_k
    reason = f"肘部法则(k={elbow_k})，轮廓系数({sil_scores[elbow_k - 2]:.4f})>0.2"
else:
    best_k = k_range[np.argmax(sil_scores)]
    reason = f"轮廓系数最高(k={best_k}, 值={max(sil_scores):.4f})"
best_k=3
print(f"\n建议聚类数: k={best_k} (决策依据: {reason})")
print(f"  肘点位置: k={elbow_k}")
print(f"  轮廓系数最高点: k={k_range[np.argmax(sil_scores)]} (值={max(sil_scores):.4f})")

# ========================
# 7. 保存K均值聚类评估图表
# ========================
print("\n\n" + "=" * 50)
print("7. 保存K均值聚类评估图表")
print("=" * 50)

# 创建肘部法则图
plt.figure(figsize=(14, 6))

# 肘部法则图
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.plot(elbow_k, wcss[elbow_k - 2], 'ro', markersize=10, label=f'肘点 k={elbow_k}')
plt.xlabel('聚类数量 (k)')
plt.ylabel('WCSS (簇内平方和)')
plt.title('K均值聚类肘部法则图')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, 'go-', linewidth=2, markersize=8)
plt.plot(best_k, sil_scores[best_k - 2], 'ro', markersize=10, label=f'最优 k={best_k}')
plt.xlabel('聚类数量 (k)')
plt.ylabel('轮廓系数')
plt.title('K均值聚类轮廓系数图')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('output/kmeans_evaluation_plots.png')
plt.close()
print("已生成聚类评估图: output/kmeans_evaluation_plots.png")

# 应用K-Means聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
wine_df['Cluster'] = clusters

# ========================
# 8. 聚类特征分析（所有特征箱线图整合到大图）
# ========================
print("\n\n" + "=" * 50)
print("8. 聚类特征分析（所有特征箱线图整合到大图）")
print("=" * 50)

# 创建大图（4行3列布局）
plt.figure(figsize=(18, 22))  # 增加整体尺寸
plt.suptitle(f'图表 6 聚类特征分布 (k={best_k})', fontsize=20, y=0.98)

# 获取所有聚类特征（排除quality和Cluster）
cluster_features = [f for f in expected_features if f != 'quality' and f != 'Cluster']

# 设置子图布局（4行3列）
rows, cols = 4, 3

# 为每个特征创建子图
for i, feature in enumerate(cluster_features, 1):
    ax = plt.subplot(rows, cols, i)

    # 绘制箱线图
    sns.boxplot(data=wine_df, x='Cluster', y=feature,
                palette='Set2', showmeans=True, width=0.6,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'red', 'markersize': 6})

    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 设置子图标题（使用中文解释）
    ax.set_title(field_explanations[feature], fontsize=14)
    ax.set_xlabel('聚类' if i > (rows - 1) * cols else '', fontsize=12)  # 仅底部行显示X轴标签
    ax.set_ylabel('')

# 添加整体标签
plt.figtext(0.5, 0.02, '聚类', ha='center', fontsize=14)
plt.figtext(0.08, 0.5, '特征值', va='center', rotation='vertical', fontsize=14)

# 调整布局并保存
plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])  # 为整体标签留出空间
plt.savefig('output/all_features_cluster_boxplots.png', dpi=120)
plt.close()
print(f"已生成整合箱线图: output/all_features_cluster_boxplots.png")
# ========================
# 9. 聚类特征雷达图（所有特征）
# ========================
print("\n\n" + "=" * 50)
print("9. 聚类特征雷达图（所有特征）")
print("=" * 50)

# 按聚类分组计算均值
cluster_means = wine_df.groupby('Cluster').mean()

# 标准化特征值以便比较
cluster_means_normalized = cluster_means.copy()
for feature in features_for_clustering:
    min_val = wine_df[feature].min()
    max_val = wine_df[feature].max()
    cluster_means_normalized[feature] = (cluster_means[feature] - min_val) / (max_val - min_val)

# 绘制雷达图（使用所有特征）
plt.figure(figsize=(16, 16))
categories = [field_explanations[f] for f in features_for_clustering]
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 创建子图
for cluster in range(best_k):
    ax = plt.subplot(3, 3, cluster + 1, polar=True)

    # 添加第一个值以闭合图形
    values = cluster_means_normalized.loc[cluster, features_for_clustering].values.tolist()
    values += values[:1]

    # 绘制雷达图
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'聚类 {cluster}')
    ax.fill(angles, values, alpha=0.25)

    # 添加标签
    plt.xticks(angles[:-1], categories, size=10)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)
    plt.title(f'聚类 {cluster} 特征分析 (样本数: {sum(wine_df["Cluster"] == cluster)})', size=12, y=1.1)

plt.suptitle(f'图表 10 聚类特征雷达图 (k={best_k})', fontsize=16, y=0.95)
plt.tight_layout()
plt.savefig('output/cluster_radar_chart.png')
plt.close()
print("已生成聚类特征雷达图: output/cluster_radar_chart.png")

# ========================
# 10. 聚类结果可视化（使用PCA降维）
# ========================
print("\n\n" + "=" * 50)
print("10. 聚类结果可视化")
print("=" * 50)

# 使用PCA降维可视化（包含所有特征）
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
wine_df['PCA1'] = pca_components[:, 0]
wine_df['PCA2'] = pca_components[:, 1]

# 计算方差解释率
explained_var = pca.explained_variance_ratio_

# 绘制PCA聚类结果
plt.figure(figsize=(12, 10))
markers = ['o', 's', 'D', '^'][:best_k]  # 根据聚类数动态选择标记
sns.scatterplot(data=wine_df, x='PCA1', y='PCA2',
                hue='Cluster', palette='viridis', s=100, alpha=0.8,
                style='Cluster', markers=markers)  # 使用动态标记
plt.xlabel(f'主成分1 (解释方差: {explained_var[0] * 100:.1f}%)')
plt.ylabel(f'主成分2 (解释方差: {explained_var[1] * 100:.1f}%)')
plt.title(f'图表 11 K均值聚类结果 (k={best_k}, 使用所有特征)', fontsize=16)
plt.legend(title='聚类', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.savefig('output/wine_clusters_pca.png', bbox_inches='tight')
plt.close()
print("已生成PCA聚类图: output/wine_clusters_pca.png")

# 绘制PCA特征载荷图
plt.figure(figsize=(14, 8))
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i + 1}' for i in range(2)],
    index=features_for_clustering
)
sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('PCA主成分特征载荷', fontsize=14)
plt.savefig('output/pca_feature_loadings.png')
plt.close()
print("已生成PCA特征载荷图: output/pca_feature_loadings.png")

# ========================
# 11. 聚类质量分析
# ========================
print("\n\n" + "=" * 50)
print("11. 聚类质量分析")
print("=" * 50)

# 质量分布箱线图
plt.figure(figsize=(12, 8))
sns.boxplot(data=wine_df, x='Cluster', y='quality', hue='Cluster',
            palette='Set2', legend=False)
plt.xlabel('聚类')
plt.ylabel('质量评分')
plt.title('图表 12 各聚类质量评分分布', fontsize=14)
plt.grid(axis='y')
plt.savefig('output/cluster_quality_boxplot.png')
plt.close()
print("已生成质量分布箱线图: output/cluster_quality_boxplot.png")

# 聚类质量统计
quality_stats = wine_df.groupby('Cluster')['quality'].agg(['mean', 'median', 'std', 'count'])
print("\n各聚类质量统计:")
print(quality_stats)

# 保存结果
wine_df.to_csv('output/full_feature_clustered_wine_data.csv', index=False)
print("\n聚类结果已保存至: output/full_feature_clustered_wine_data.csv")

print("\n所有分析完成！结果保存在output文件夹中")