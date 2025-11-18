import streamlit as st


def render_functional_clustering_params():
    """
    渲染功能聚类参数设置界面
    
    返回:
        dict: 包含所有参数的字典
    """
    params = {}
    
    with st.expander("功能聚类参数设置", expanded=False):
        # 基本参数
        params['n_components'] = st.slider("聚类数量（组件数）", min_value=2, max_value=10, value=3, step=1)
        
        # 均值函数类型
        mean_types = ['None', 'power_equation', 'logistic_growth', 'linear_equation', 'fourier_series1', 'fourier_series2']
        mean_type = st.selectbox(
            "均值函数类型",
            mean_types,
            index=1,  # 默认使用 power_equation
            key="func_mean_type",
            help="None: 不使用函数建模均值\n"
                 "power_equation: 幂函数方程\n"
                 "logistic_growth: 逻辑增长方程\n"
                 "linear_equation: 线性方程\n"
                 "fourier_series1: 傅里叶级数（4参数）\n"
                 "fourier_series2: 傅里叶级数（6参数）"
        )
        params['mean_type'] = mean_type
        
        # 协方差类型
        covariance_types = ['SAD1_tied', 'AR1_tied', 'full', 'tied', 'diag', 'spherical', 'SAD1_full', 'AR1_full']
        covariance_type = st.selectbox(
            "协方差类型",
            covariance_types,
            index=0,  # 默认使用 SAD1_tied
            key="func_covariance_type",
            help="SAD1_tied: SAD1结构，所有成分共享\n"
                 "AR1_tied: AR1结构，所有成分共享\n"
                 "full: 每个成分有独立的完整协方差矩阵\n"
                 "tied: 所有成分共享相同的协方差矩阵\n"
                 "diag: 每个成分有独立的对角协方差矩阵\n"
                 "spherical: 每个成分有独立的球形协方差矩阵\n"
                 "SAD1_full: SAD1结构，每个成分独立\n"
                 "AR1_full: AR1结构，每个成分独立"
        )
        params['covariance_type'] = covariance_type
        
        # 优化参数
        st.subheader("优化参数")
        params['max_iter'] = st.slider("最大迭代次数（EM算法）", min_value=10, max_value=200, value=50, step=10)
        
        params['m_step_epochs'] = st.slider("M步优化迭代次数", min_value=10, max_value=200, value=50, step=10,
                                           help="每次M步中优化参数的迭代次数")
        
        params['learning_rate'] = st.slider("学习率", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
        
        params['tol'] = st.slider("收敛阈值", min_value=1e-5, max_value=1e-2, value=1e-3, step=1e-4, format="%.5f",
                                 help="EM算法收敛的阈值")
        
        random_state = st.number_input("随机种子 (None表示随机)", 
                                      min_value=None, max_value=None, 
                                      value=None, step=1,
                                      key="func_random_state")
        if random_state is not None:
            params['random_state'] = int(random_state)
        else:
            params['random_state'] = None
        
        # 初始化方法
        init_methods = ['kmeans', 'random']
        params['init_params'] = st.selectbox(
            "初始化方法",
            init_methods,
            index=0,
            key="func_init_params",
            help="kmeans: 使用KMeans初始化\n"
                 "random: 随机初始化"
        )
        
        # 数据转换选项
        st.subheader("数据选项")
        params['trans_data'] = st.checkbox(
            "对数转换数据 (log10(x+1))",
            value=False,
            key="func_trans_data",
            help="是否对数据进行对数转换"
        )
    
    return params

