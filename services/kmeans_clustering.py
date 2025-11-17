import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# 支持绘图中文
plt.rcParams['font.family'] = ['SimHei']   # 设置中文黑体字体
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号


def perform_kmeans_clustering(data, n_clusters=3, n_init='auto', random_state=None):
    """
    执行 KMeans 聚类
    
    参数:
        data: pandas DataFrame 或 numpy array，输入数据
        n_clusters: int，聚类数量，默认3
        n_init: int 或 'auto'，初始化次数，默认'auto'
        random_state: int，随机种子，默认None
    
    返回:
        kmeans: 训练好的 KMeans 模型
    """
    # 转换为 numpy array
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    
    # 创建并训练 KMeans 模型
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(X)
    
    return kmeans


def plot_kmeans_clustering(kmeans, X_train, feature_names=None, 
                           xlim=None, ylim=None, plot_step=0.02, 
                           cmap_light='Pastel2'):
    """
    可视化 KMeans 聚类结果
    
    参数:
        kmeans: 训练好的 KMeans 模型
        X_train: numpy array，训练数据
        feature_names: list，特征名称列表，默认None
        xlim: tuple，x轴范围，默认None（自动计算）
        ylim: tuple，y轴范围，默认None（自动计算）
        plot_step: float，网格步长，默认0.02
        cmap_light: str，颜色映射，默认'Pastel2'
    
    返回:
        fig: matplotlib figure 对象
    """
    # 确保 X_train 是 numpy array
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    # 自动计算坐标范围
    if xlim is None:
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    else:
        x_min, x_max = xlim
    
    if ylim is None:
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    else:
        y_min, y_max = ylim
    
    # 生成网格数据
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int((x_max - x_min) / plot_step + 1)),
        np.linspace(y_min, y_max, int((y_max - y_min) / plot_step + 1))
    )
    
    # 使用KMeans模型对网格中的点进行预测
    # 确保数组是 C 连续的
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points = np.ascontiguousarray(grid_points)
    Z = kmeans.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制区域
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    
    # 绘制样本数据点
    plt.scatter(X_train[:, 0], X_train[:, 1], 
                color=np.array([0, 68, 138])/255., 
                alpha=1.0, linewidth=1, edgecolor=[1, 1, 1], s=50)
    
    # 绘制决策边界
    plt.contour(xx, yy, Z, levels=range(kmeans.n_clusters), 
                colors=np.array([0, 68, 138])/255., linewidths=1.5)
    
    # 绘制聚类中心
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                marker="x", s=200, linewidths=2, color="k", zorder=10)
    
    # 设置坐标轴
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 设置标签
    if feature_names is not None:
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
    else:
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
    
    # 添加网格
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig


def perform_kmeans_with_iterations(data, n_clusters=3, max_iter=300, random_state=None, 
                                   return_history=True):
    """
    执行 KMeans 聚类并记录每次迭代的状态
    
    参数:
        data: pandas DataFrame 或 numpy array，输入数据
        n_clusters: int，聚类数量，默认3
        max_iter: int，最大迭代次数，默认300
        random_state: int，随机种子，默认None
        return_history: bool，是否返回迭代历史，默认True
    
    返回:
        kmeans: 训练好的 KMeans 模型
        history: list，每次迭代的状态（中心点、标签等）
    """
    # 转换为 numpy array
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    
    # 初始化中心点
    if random_state is not None:
        np.random.seed(random_state)
    
    # 随机选择初始中心点
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
    
    history = []
    
    for iteration in range(max_iter):
        # 计算每个点到最近中心的距离，分配标签
        labels = pairwise_distances_argmin(X, centroids)
        
        # 保存当前迭代状态
        if return_history:
            history.append({
                'iteration': iteration,
                'centroids': centroids.copy(),
                'labels': labels.copy()
            })
        
        # 计算新的中心点
        new_centroids = np.array([X[labels == i].mean(axis=0) 
                                 for i in range(n_clusters)])
        
        # 检查是否收敛
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
        
        centroids = new_centroids
    
    # 创建 KMeans 对象并正确初始化
    # 使用最终的中心点作为初始值，只运行一次迭代（实际上已经收敛了）
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1, 
                   init=centroids, random_state=random_state)
    # 调用 fit 来初始化所有内部属性（包括 _n_threads 等）
    kmeans.fit(X)
    # 然后设置我们计算的值（因为我们已经手动计算了，所以覆盖 fit 的结果）
    kmeans.cluster_centers_ = centroids
    kmeans.labels_ = labels
    kmeans.n_iter_ = iteration + 1
    
    if return_history:
        return kmeans, history
    else:
        return kmeans


def plot_kmeans_iteration(centroids, labels, X_train, iteration, feature_names=None,
                          xlim=None, ylim=None, plot_step=0.02, cmap_light='Pastel2',
                          show_centroid_path=False, previous_centroids=None):
    """
    可视化 KMeans 单次迭代的结果
    
    参数:
        centroids: numpy array，当前迭代的中心点
        labels: numpy array，当前迭代的标签
        X_train: numpy array，训练数据
        iteration: int，当前迭代次数
        feature_names: list，特征名称列表，默认None
        xlim: tuple，x轴范围，默认None（自动计算）
        ylim: tuple，y轴范围，默认None（自动计算）
        plot_step: float，网格步长，默认0.02
        cmap_light: str，颜色映射，默认'Pastel2'
        show_centroid_path: bool，是否显示中心点移动路径，默认False
        previous_centroids: numpy array，上一次迭代的中心点，默认None
    
    返回:
        fig: matplotlib figure 对象
    """
    # 确保 X_train 是 numpy array
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    # 自动计算坐标范围
    if xlim is None:
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    else:
        x_min, x_max = xlim
    
    if ylim is None:
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    else:
        y_min, y_max = ylim
    
    # 生成网格数据
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int((x_max - x_min) / plot_step + 1)),
        np.linspace(y_min, y_max, int((y_max - y_min) / plot_step + 1))
    )
    
    # 计算每个网格点到最近中心的距离
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # 确保数组是 C 连续的
    grid_points = np.ascontiguousarray(grid_points)
    centroids = np.ascontiguousarray(centroids)
    grid_labels = pairwise_distances_argmin(grid_points, centroids)
    Z = grid_labels.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制区域
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    
    # 绘制样本数据点，根据标签着色
    try:
        # 尝试使用新版本 matplotlib
        cmap = plt.colormaps[cmap_light]
    except (AttributeError, KeyError):
        # 使用旧版本或直接使用 cmap
        try:
            cmap = plt.get_cmap(cmap_light)
        except AttributeError:
            cmap = plt.cm.get_cmap(cmap_light)
    colors = cmap(np.linspace(0, 1, len(np.unique(labels))))
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                   color=colors[i], alpha=0.8, linewidth=1, 
                   edgecolor='white', s=50, label=f'Cluster {label}')
    
    # 显示中心点移动路径
    if show_centroid_path and previous_centroids is not None:
        for i in range(len(centroids)):
            plt.plot([previous_centroids[i, 0], centroids[i, 0]],
                    [previous_centroids[i, 1], centroids[i, 1]],
                    'k--', alpha=0.5, linewidth=1)
    
    # 绘制决策边界
    plt.contour(xx, yy, Z, levels=range(len(centroids)), 
                colors=np.array([0, 68, 138])/255., linewidths=1.5)
    
    # 绘制聚类中心
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                marker="x", s=300, linewidths=3, color="red", 
                zorder=10, label='Centroids')
    
    # 如果显示路径，也绘制上一次的中心点
    if show_centroid_path and previous_centroids is not None:
        plt.scatter(previous_centroids[:, 0], previous_centroids[:, 1], 
                   marker="o", s=100, linewidths=2, color="orange", 
                   zorder=9, alpha=0.6, label='Previous Centroids')
    
    # 设置坐标轴
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 设置标签
    if feature_names is not None:
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
    else:
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
    
    # 添加标题
    plt.title(f'KMeans 迭代 {iteration + 1}', fontsize=14, fontweight='bold')
    
    # 添加图例
    plt.legend(loc='upper right', fontsize=10)
    
    # 添加网格
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig
