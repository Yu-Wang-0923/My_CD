"""
功能聚类服务
基于参考代码实现的功能聚类功能
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 延迟导入 torch，避免在导入模块时就出错
# torch 只在需要时才导入


def check_torch_available():
    """检查 PyTorch 是否可用"""
    try:
        import torch
        # 尝试执行一个简单的操作来验证 torch 是否正常工作
        _ = torch.tensor([1.0])
        return True
    except (ImportError, OSError, RuntimeError) as e:
        return False


def perform_functional_clustering_simple(data, n_components=3, mean_type='power_equation', 
                                        covariance_type='SAD1_tied', max_iter=50, 
                                        random_state=None, verbose=True):
    """
    简化的功能聚类实现（如果完整版本不可用）
    
    参数:
        data: pandas DataFrame 或 numpy array，输入数据
        n_components: int，聚类数量
        mean_type: str，均值函数类型
        covariance_type: str，协方差类型
        max_iter: int，最大迭代次数
        random_state: int，随机种子
        verbose: bool，是否显示详细信息
    
    返回:
        dict: 包含聚类结果的字典
    """
    # 转换为 numpy array
    if isinstance(data, pd.DataFrame):
        X = data.values
        columns = data.columns.tolist()
        index = data.index.tolist()
    else:
        X = data
        columns = None
        index = None
    
    # 这里应该调用实际的 FunClu 类
    # 由于参考代码较复杂，这里提供一个占位实现
    # 实际使用时需要集成完整的 FunClu 类
    
    # 临时使用 KMeans 作为占位
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_components, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    result = {
        'labels': labels,
        'n_components': n_components,
        'mean_type': mean_type,
        'covariance_type': covariance_type,
        'data': X,
        'columns': columns,
        'index': index,
        'model_type': 'simplified'  # 标记为简化版本
    }
    
    return result


def perform_functional_clustering(data, n_components=3, mean_type='power_equation', 
                                 covariance_type='SAD1_tied', max_iter=50, 
                                 random_state=None, verbose=True, times=None, params=None):
    """
    执行功能聚类
    
    参数:
        data: pandas DataFrame 或 numpy array，输入数据
        n_components: int，聚类数量
        mean_type: str，均值函数类型 ('None', 'power_equation', 'logistic_growth', 'linear_equation', 'fourier_series1', 'fourier_series2')
        covariance_type: str，协方差类型 ('SAD1_tied', 'AR1_tied', 'full', 'tied', 'diag', 'spherical', 'SAD1_full', 'AR1_full')
        max_iter: int，最大迭代次数
        random_state: int，随机种子
        verbose: bool，是否显示详细信息
        times: array-like，时间点（如果为None，将自动生成）
    
    返回:
        dict: 包含聚类结果的字典
    """
    # 检查 PyTorch 是否可用
    if not check_torch_available():
        # 如果 PyTorch 不可用，使用简化版本
        if verbose:
            print("PyTorch 未安装，使用简化版本的功能聚类")
        return perform_functional_clustering_simple(
            data, n_components, mean_type, covariance_type, max_iter, random_state, verbose
        )
    
    # 尝试使用完整版本（如果 FunClu 类可用）
    try:
        from services.clustering.functional_clustering_model import FunClu
        
        # 转换为 DataFrame（如果需要）
        if isinstance(data, pd.DataFrame):
            data_df = data
        else:
            data_df = pd.DataFrame(data)
        
        # 从参数中获取学习率和其他参数
        learning_rate = params.get('learning_rate', 0.01) if params is not None else 0.01
        tol = params.get('tol', 1e-3) if params is not None else 1e-3
        trans_data = params.get('trans_data', False) if params is not None else False
        m_step_epochs = params.get('m_step_epochs', 50) if params is not None else 50
        
        # 创建模型
        model = FunClu(
            K=n_components,
            mean_type=mean_type,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
            lr=learning_rate,
            tol=tol
        )
        
        # 拟合模型
        fit_result = model.fit(
            data_df, 
            times=times, 
            trans_data=trans_data, 
            max_iter=max_iter, 
            verbose=verbose,
            verbose_interval=10 if verbose else 100,
            m_step_epochs=m_step_epochs
        )
        
        if fit_result == -1:
            # 如果拟合失败，使用简化版本
            if verbose:
                print("功能聚类拟合失败，使用简化版本")
            return perform_functional_clustering_simple(
                data, n_components, mean_type, covariance_type, max_iter, random_state, verbose
            )
        
        # 获取聚类标签
        labels = np.argmax(model.parameters["resp"].cpu().numpy(), 1)
        
        # 准备结果
        result = {
            'labels': labels,
            'n_components': n_components,
            'mean_type': mean_type,
            'covariance_type': covariance_type,
            'data': data_df.values if isinstance(data_df, pd.DataFrame) else data_df,
            'columns': data_df.columns.tolist() if isinstance(data_df, pd.DataFrame) else None,
            'index': data_df.index.tolist() if isinstance(data_df, pd.DataFrame) else None,
            'model': model,
            'pars_means': model.parameters['pars_means'].detach().cpu().numpy() if 'pars_means' in model.parameters else None,
            'pars_covariance': model.parameters['pars_covariance'].detach().cpu().numpy() if 'pars_covariance' in model.parameters else None,
            'BIC': model.bic() if hasattr(model, 'bic') and model.is_fit else None,
            'model_type': 'full'
        }
        
        return result
        
    except ImportError as e:
        # FunClu 类不可用，使用简化版本
        if verbose:
            print(f"FunClu 类未找到 ({str(e)})，使用简化版本的功能聚类")
        return perform_functional_clustering_simple(
            data, n_components, mean_type, covariance_type, max_iter, random_state, verbose
        )
    except Exception as e:
        # 其他错误，使用简化版本
        if verbose:
            print(f"功能聚类执行出错: {str(e)}，使用简化版本")
        import traceback
        traceback.print_exc()
        return perform_functional_clustering_simple(
            data, n_components, mean_type, covariance_type, max_iter, random_state, verbose
        )


def plot_functional_clustering_results(result, feature_names=None):
    """
    可视化功能聚类结果
    
    参数:
        result: dict，功能聚类结果
        feature_names: list，特征名称列表
    
    返回:
        matplotlib figure 对象
    """
    import matplotlib.pyplot as plt
    from components.utils.font_config import configure_matplotlib_chinese
    configure_matplotlib_chinese()
    
    labels = result['labels']
    data = result['data']
    n_components = result['n_components']
    
    # 如果是2D数据，绘制散点图
    if data.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 为每个聚类分配颜色
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        
        for k in range(n_components):
            mask = labels == k
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=[colors[k]], label=f'Cluster {k+1}', 
                      alpha=0.6, s=50)
        
        if feature_names:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
        else:
            ax.set_xlabel("特征 1")
            ax.set_ylabel("特征 2")
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("功能聚类结果")
        plt.tight_layout()
        
        return fig
    else:
        # 多维数据，显示聚类标签分布
        fig, ax = plt.subplots(figsize=(10, 6))
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax.bar(unique_labels + 1, counts, alpha=0.7)
        ax.set_xlabel("聚类")
        ax.set_ylabel("样本数量")
        ax.set_title("功能聚类结果 - 各聚类的样本数量")
        ax.set_xticks(unique_labels + 1)
        plt.tight_layout()
        
        return fig

