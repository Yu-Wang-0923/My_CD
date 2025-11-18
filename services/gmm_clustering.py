import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

# 支持绘图中文
plt.rcParams['font.family'] = ['SimHei']   # 设置中文黑体字体
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号

# 设置绘图样式（参考用户提供的代码）
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


def perform_gmm_clustering(data, n_components=3, covariance_type='full', 
                          max_iter=100, random_state=None):
    """
    执行 GMM 聚类
    
    参数:
        data: pandas DataFrame 或 numpy array，输入数据
        n_components: int，聚类数量（混合成分数），默认3
        covariance_type: str，协方差类型，默认'full'
                        可选: 'full', 'tied', 'diag', 'spherical'
        max_iter: int，最大迭代次数，默认100
        random_state: int，随机种子，默认None
    
    返回:
        gmm: 训练好的 GaussianMixture 模型
    """
    # 转换为 numpy array
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    
    # 创建并训练 GMM 模型
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state
    )
    gmm.fit(X)
    
    return gmm


def make_ellipses(gmm, ax, n_components, rgb_colors):
    """
    绘制 GMM 的椭圆（表示协方差矩阵）
    
    参数:
        gmm: 训练好的 GaussianMixture 模型
        ax: matplotlib axes 对象
        n_components: int，聚类数量
        rgb_colors: numpy array，颜色数组 (n_components, 3)
    """
    for j in range(n_components):
        # 根据协方差类型获取协方差矩阵
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[j]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[j])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) 
            covariances = covariances * gmm.covariances_[j]
        
        # 用奇异值分解完成特征值分解
        U, S, V_T = np.linalg.svd(covariances)
        # 计算长轴、短轴长度
        major, minor = 2 * np.sqrt(S)
        
        # 计算椭圆长轴旋转角度
        angle = np.arctan2(U[1, 0], U[0, 0])
        angle = 180 * angle / np.pi
        
        # 多元高斯分布中心
        ax.plot(gmm.means_[j, 0], gmm.means_[j, 1],
                color='k', marker='x', markersize=10)
        
        # 绘制半长轴向量
        ax.quiver(gmm.means_[j, 0], gmm.means_[j, 1],
                 U[0, 0], U[1, 0], scale=5/major)
        
        # 绘制半短轴向量
        ax.quiver(gmm.means_[j, 0], gmm.means_[j, 1],
                 U[0, 1], U[1, 1], scale=5/minor)
        
        # 绘制椭圆（3个不同尺度的椭圆）
        for scale in np.array([3, 2, 1]):
            ell = Ellipse(
                xy=(gmm.means_[j, 0], gmm.means_[j, 1]),
                width=scale * major,
                height=scale * minor,
                angle=angle,
                color=rgb_colors[j, :],
                alpha=0.18
            )
            ax.add_artist(ell)


def plot_gmm_clustering(gmm, X_train, feature_names=None,
                        xlim=None, ylim=None, plot_step=0.02):
    """
    可视化 GMM 聚类结果
    
    参数:
        gmm: 训练好的 GaussianMixture 模型
        X_train: numpy array，训练数据
        feature_names: list，特征名称列表，默认None
        xlim: tuple，x轴范围，默认None（自动计算）
        ylim: tuple，y轴范围，默认None（自动计算）
        plot_step: float，网格步长，默认0.02
    
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
    x1_array = np.linspace(x_min, x_max, int((x_max - x_min) / plot_step + 1))
    x2_array = np.linspace(y_min, y_max, int((y_max - y_min) / plot_step + 1))
    xx1, xx2 = np.meshgrid(x1_array, x2_array)
    
    # 使用 GMM 模型对网格中的点进行预测
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    grid_points = np.ascontiguousarray(grid_points)
    Z = gmm.predict(grid_points)
    Z = Z.reshape(xx1.shape)
    
    # 创建颜色映射
    n_components = gmm.n_components
    rgb = np.array([[255, 51, 0], 
                    [0, 153, 255],
                    [138, 138, 138],
                    [255, 200, 0],
                    [0, 200, 100],
                    [200, 0, 255],
                    [255, 100, 100],
                    [100, 255, 100],
                    [100, 100, 255],
                    [255, 150, 50]])[:n_components] / 255.0
    cmap_bold = ListedColormap(rgb)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 6))
    
    # 子图1: 椭圆和向量可视化
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x=X_train[:, 0], y=X_train[:, 1],
               color=np.array([0, 68, 138])/255.,
               alpha=1.0,
               linewidth=1, edgecolor=[1, 1, 1], s=50)
    # 绘制椭圆和向量
    make_ellipses(gmm, ax1, n_components, rgb)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    if feature_names:
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
    else:
        ax1.set_xlabel("特征 1")
        ax1.set_ylabel("特征 2")
    ax1.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("GMM 聚类结果（椭圆表示协方差）")
    
    # 子图2: 等高线图
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(xx1, xx2, Z, cmap=cmap_bold, alpha=0.18)
    ax2.contour(xx1, xx2, Z, levels=range(n_components),
               colors=[np.array([0, 68, 138])/255.])
    ax2.scatter(x=X_train[:, 0], y=X_train[:, 1],
               color=np.array([0, 68, 138])/255.,
               alpha=1.0,
               linewidth=1, edgecolor=[1, 1, 1], s=50)
    centroids = gmm.means_
    ax2.scatter(centroids[:, 0], centroids[:, 1],
               marker="x", s=100, linewidths=1.5,
               color="k")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    if feature_names:
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
    else:
        ax2.set_xlabel("特征 1")
        ax2.set_ylabel("特征 2")
    ax2.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title("GMM 聚类区域划分")
    
    plt.tight_layout()
    
    return fig


def calculate_gmm_aic_bic(gmm_model, X):
    """
    计算 GMM 的 AIC 和 BIC
    
    参数:
        gmm_model: 训练好的 GaussianMixture 模型
        X: numpy array，训练数据
    
    返回:
        aic: float，AIC 值
        bic: float，BIC 值
        log_likelihood: float，对数似然值
    """
    # 转换为 numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # 计算对数似然
    log_likelihood = gmm_model.score(X)
    
    # 计算参数数量
    n_samples, n_features = X.shape
    n_components = gmm_model.n_components
    
    # 计算参数数量 k
    # 对于 GMM:
    # - 每个成分的均值: n_components * n_features
    # - 每个成分的协方差: 取决于 covariance_type
    #   - 'full': n_components * n_features * (n_features + 1) / 2
    #   - 'tied': n_features * (n_features + 1) / 2
    #   - 'diag': n_components * n_features
    #   - 'spherical': n_components
    # - 混合权重: n_components - 1 (因为和为1)
    
    if gmm_model.covariance_type == 'full':
        cov_params = n_components * n_features * (n_features + 1) / 2
    elif gmm_model.covariance_type == 'tied':
        cov_params = n_features * (n_features + 1) / 2
    elif gmm_model.covariance_type == 'diag':
        cov_params = n_components * n_features
    elif gmm_model.covariance_type == 'spherical':
        cov_params = n_components
    
    mean_params = n_components * n_features
    weight_params = n_components - 1
    k = mean_params + cov_params + weight_params
    
    # 计算 AIC 和 BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n_samples) - 2 * log_likelihood
    
    return aic, bic, log_likelihood


def plot_gmm_elbow_method(data, max_components=10, min_components=1, 
                          covariance_type='full', max_iter=100, random_state=None):
    """
    绘制 GMM 肘部法则图，包括对数似然、AIC 和 BIC
    
    参数:
        data: pandas DataFrame 或 numpy array，输入数据
        max_components: int，最大组件数量，默认10
        min_components: int，最小组件数量，默认1
        covariance_type: str，协方差类型，默认'full'
        max_iter: int，最大迭代次数，默认100
        random_state: int，随机种子，默认None
    
    返回:
        fig: matplotlib figure 对象
        results: dict，包含不同组件数量的结果
    """
    # 转换为 numpy array
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    
    n_samples, n_features = X.shape
    n_components_range = range(min_components, max_components + 1)
    
    log_likelihood_values = []
    aic_values = []
    bic_values = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state
        )
        gmm.fit(X)
        
        aic, bic, log_likelihood = calculate_gmm_aic_bic(gmm, X)
        log_likelihood_values.append(log_likelihood)
        aic_values.append(aic)
        bic_values.append(bic)
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 对数似然
    axes[0].plot(n_components_range, log_likelihood_values, marker='x', linewidth=2, markersize=8, color='b')
    axes[0].set_xlabel('组件数量 (k)', fontsize=12)
    axes[0].set_ylabel('对数似然', fontsize=12)
    axes[0].set_title('对数似然', fontsize=14, fontweight='bold')
    axes[0].grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    axes[0].set_xticks(n_components_range)
    # 标记最大对数似然值
    max_ll_idx = np.argmax(log_likelihood_values)
    axes[0].plot(n_components_range[max_ll_idx], log_likelihood_values[max_ll_idx], 'b*', 
                markersize=20, label=f'最大对数似然: k={n_components_range[max_ll_idx]}')
    axes[0].legend()
    
    # AIC
    axes[1].plot(n_components_range, aic_values, marker='x', linewidth=2, markersize=8, color='r')
    axes[1].set_xlabel('组件数量 (k)', fontsize=12)
    axes[1].set_ylabel('AIC', fontsize=12)
    axes[1].set_title('AIC (赤池信息准则)', fontsize=14, fontweight='bold')
    axes[1].grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    axes[1].set_xticks(n_components_range)
    # 标记最小AIC值
    min_aic_idx = np.argmin(aic_values)
    axes[1].plot(n_components_range[min_aic_idx], aic_values[min_aic_idx], 'r*', 
                markersize=20, label=f'最小 AIC: k={n_components_range[min_aic_idx]}')
    axes[1].legend()
    
    # BIC
    axes[2].plot(n_components_range, bic_values, marker='x', linewidth=2, markersize=8, color='g')
    axes[2].set_xlabel('组件数量 (k)', fontsize=12)
    axes[2].set_ylabel('BIC', fontsize=12)
    axes[2].set_title('BIC (贝叶斯信息准则)', fontsize=14, fontweight='bold')
    axes[2].grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
    axes[2].set_xticks(n_components_range)
    # 标记最小BIC值
    min_bic_idx = np.argmin(bic_values)
    axes[2].plot(n_components_range[min_bic_idx], bic_values[min_bic_idx], 'g*', 
                markersize=20, label=f'最小 BIC: k={n_components_range[min_bic_idx]}')
    axes[2].legend()
    
    plt.tight_layout()
    
    # 准备结果字典
    results = {
        'n_components_range': list(n_components_range),
        'log_likelihood': log_likelihood_values,
        'aic': aic_values,
        'bic': bic_values,
        'optimal_k_aic': n_components_range[min_aic_idx],
        'optimal_k_bic': n_components_range[min_bic_idx],
        'optimal_k_ll': n_components_range[max_ll_idx]
    }
    
    return fig, results

