import streamlit as st


def render_gmm_params():
    """
    渲染 GMM 参数设置界面
    
    返回:
        dict: 包含所有参数的字典
    """
    params = {}
    
    with st.expander("GMM 参数设置", expanded=False):
        # 可控制的参数
        params['n_components'] = st.slider("聚类数量（混合成分数）", min_value=2, max_value=10, value=3, step=1)
        
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        covariance_type = st.radio(
            "协方差类型",
            covariance_types,
            index=0,
            help="full: 每个成分有独立的完整协方差矩阵\n"
                 "tied: 所有成分共享相同的协方差矩阵\n"
                 "diag: 每个成分有独立的对角协方差矩阵\n"
                 "spherical: 每个成分有独立的球形协方差矩阵"
        )
        params['covariance_type'] = covariance_type
        
        params['max_iter'] = st.slider("最大迭代次数", min_value=10, max_value=500, value=100, step=10)
        
        random_state = st.number_input("随机种子 (None表示随机)", 
                                      min_value=None, max_value=None, 
                                      value=None, step=1)
        if random_state is not None:
            params['random_state'] = int(random_state)
        else:
            params['random_state'] = None
        
        # 可视化参数
        st.subheader("可视化参数")
        params['plot_step'] = st.slider("网格步长", min_value=0.01, max_value=0.1, 
                                  value=0.02, step=0.01)
    
    return params

