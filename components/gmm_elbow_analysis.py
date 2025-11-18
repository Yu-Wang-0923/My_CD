import streamlit as st
import pandas as pd
from services.gmm_clustering import plot_gmm_elbow_method


def render_gmm_elbow_analysis():
    """
    渲染 GMM 肘部法则分析界面
    """
    st.header("肘部法则分析 (AIC & BIC)")
    
    with st.expander("肘部法则参数设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_components_elbow = st.slider("最小组件数量", min_value=1, max_value=5, value=1, step=1, key="gmm_min_components_elbow")
            max_components_elbow = st.slider("最大组件数量", min_value=3, max_value=15, value=10, step=1, key="gmm_max_components_elbow")
        with col2:
            covariance_type_elbow = st.selectbox(
                "协方差类型", 
                options=['full', 'tied', 'diag', 'spherical'], 
                index=0, 
                key="gmm_covariance_type_elbow",
                help="full: 每个成分有独立的完整协方差矩阵\n"
                     "tied: 所有成分共享相同的协方差矩阵\n"
                     "diag: 每个成分有独立的对角协方差矩阵\n"
                     "spherical: 每个成分有独立的球形协方差矩阵"
            )
            max_iter_elbow = st.slider("最大迭代次数", min_value=10, max_value=500, value=100, step=10, key="gmm_max_iter_elbow")
        
        random_state_elbow = st.number_input("随机种子 (None表示随机)", 
                                             min_value=None, max_value=None, 
                                             value=None, step=1, key="gmm_random_state_elbow")
        if random_state_elbow is not None:
            random_state_elbow = int(random_state_elbow)
        else:
            random_state_elbow = None
    
    if st.button("执行肘部法则分析", key="gmm_run_elbow_button"):
        with st.spinner("正在计算不同组件数量的指标..."):
            fig_elbow, results_elbow = plot_gmm_elbow_method(
                st.session_state.gmm_clustering_data,
                min_components=min_components_elbow,
                max_components=max_components_elbow,
                covariance_type=covariance_type_elbow,
                max_iter=max_iter_elbow,
                random_state=random_state_elbow
            )
            st.session_state.gmm_elbow_results = results_elbow
            st.pyplot(fig_elbow)
            
            # 显示最优组件数量
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AIC 推荐组件数量", results_elbow['optimal_k_aic'])
            with col2:
                st.metric("BIC 推荐组件数量", results_elbow['optimal_k_bic'])
            with col3:
                st.metric("最大对数似然组件数量", results_elbow['optimal_k_ll'])
            
            # 显示详细数据表
            with st.expander("查看详细数据", expanded=False):
                results_df = pd.DataFrame({
                    '组件数量 (k)': results_elbow['n_components_range'],
                    '对数似然': results_elbow['log_likelihood'],
                    'AIC': results_elbow['aic'],
                    'BIC': results_elbow['bic']
                })
                st.dataframe(results_df, use_container_width=True)

