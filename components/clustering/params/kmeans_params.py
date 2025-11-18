import streamlit as st


def render_kmeans_params():
    """
    渲染 KMeans 参数设置界面
    
    返回:
        dict: 包含所有参数的字典
    """
    params = {}
    
    with st.expander("KMeans 参数设置", expanded=False):
        # 可控制的参数
        params['n_clusters'] = st.slider("聚类数量", min_value=2, max_value=10, value=3, step=1)
        
        n_init = st.selectbox("初始化次数", options=['auto', 10, 20, 50, 100], index=0)
        if n_init == 'auto':
            params['n_init'] = 'auto'
        else:
            params['n_init'] = int(n_init)
        
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
        params['cmap_light'] = st.selectbox("颜色映射", 
                                     options=['Pastel2', 'Pastel1', 'Set3', 'Set2', 'Set1'],
                                     index=0)
        
        # 迭代参数
        st.subheader("迭代参数")
        params['max_iter'] = st.slider("最大迭代次数", min_value=1, max_value=50, value=10, step=1)
        params['show_iteration'] = st.checkbox("显示迭代过程", value=True)
        params['show_centroid_path'] = st.checkbox("显示中心点移动路径", value=True)
    
    return params

