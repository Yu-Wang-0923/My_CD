import streamlit as st
import pandas as pd
from services.kmeans_clustering import plot_elbow_method


def render_elbow_analysis():
    """
    渲染肘部法则分析界面
    """
    st.header("肘部法则分析 (AIC & BIC)")
    
    with st.expander("肘部法则参数设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_clusters_elbow = st.slider("最小聚类数量", min_value=1, max_value=5, value=1, step=1, key="min_clusters_elbow")
            max_clusters_elbow = st.slider("最大聚类数量", min_value=3, max_value=15, value=10, step=1, key="max_clusters_elbow")
        with col2:
            init_method = st.selectbox("初始化方法", options=['random', 'k-means++'], index=0, key="init_method")
            n_init_elbow = st.selectbox("初始化次数", options=['auto', 10, 20, 50, 100], index=0, key="n_init_elbow")
        
        if n_init_elbow == 'auto':
            n_init_elbow_value = 'auto'
        else:
            n_init_elbow_value = int(n_init_elbow)
        
        random_state_elbow = st.number_input("随机种子 (None表示随机)", 
                                             min_value=None, max_value=None, 
                                             value=None, step=1, key="random_state_elbow")
        if random_state_elbow is not None:
            random_state_elbow = int(random_state_elbow)
        else:
            random_state_elbow = None
    
    if st.button("执行肘部法则分析", key="run_elbow_button"):
        with st.spinner("正在计算不同聚类数量的指标..."):
            fig_elbow, results_elbow = plot_elbow_method(
                st.session_state.clustering_data,
                min_clusters=min_clusters_elbow,
                max_clusters=max_clusters_elbow,
                n_init=n_init_elbow_value,
                init=init_method,
                random_state=random_state_elbow
            )
            st.session_state.elbow_results = results_elbow
            st.pyplot(fig_elbow)
            
            # 显示最优聚类数量
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AIC 推荐聚类数量", results_elbow['optimal_k_aic'])
            with col2:
                st.metric("BIC 推荐聚类数量", results_elbow['optimal_k_bic'])
            
            # 显示详细数据表
            with st.expander("查看详细数据", expanded=False):
                results_df = pd.DataFrame({
                    '聚类数量 (k)': results_elbow['k_range'],
                    'SSE': results_elbow['sse'],
                    'AIC': results_elbow['aic'],
                    'BIC': results_elbow['bic']
                })
                st.dataframe(results_df, use_container_width=True)

