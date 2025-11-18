import streamlit as st
import pandas as pd
import numpy as np
from services.gmm_clustering import plot_gmm_clustering, calculate_gmm_aic_bic
from components.clustering_utils import get_viz_data


def get_gmm_viz_data():
    """
    获取用于 GMM 可视化的数据（仅选定的两个特征）
    GMM 聚类结果可视化总是使用标准化后的数据（如果数据被标准化了）
    
    返回:
        pandas DataFrame: 用于可视化的数据
    """
    selected_features = st.session_state.gmm_selected_features
    
    # 检查 GMM 数据是否被标准化
    gmm_is_normalized = False
    if 'gmm_data_transformation_method' in st.session_state:
        method = st.session_state.gmm_data_transformation_method
        gmm_is_normalized = (method != "不转换")
    
    # GMM 聚类结果可视化总是使用标准化后的数据
    if gmm_is_normalized and st.session_state.gmm_clustering_data is not None:
        # 使用标准化后的数据可视化
        if st.session_state.gmm_clustering_data.shape[1] > 2:
            viz_data = st.session_state.gmm_clustering_data[selected_features]
        else:
            viz_data = st.session_state.gmm_clustering_data
    else:
        # 如果数据没有被标准化，使用原始数据
        if st.session_state.gmm_clustering_data.shape[1] > 2:
            # 如果使用了所有特征进行聚类，需要提取选定的两个特征用于可视化
            if 'gmm_transformed_df' in st.session_state and st.session_state.gmm_transformed_df is not None:
                viz_data = st.session_state.gmm_transformed_df[selected_features]
            elif 'gmm_uploaded_df' in st.session_state and st.session_state.gmm_uploaded_df is not None:
                viz_data = st.session_state.gmm_uploaded_df[selected_features]
            else:
                viz_data = st.session_state.gmm_clustering_data[selected_features]
        else:
            # 如果只使用了两个特征，直接使用聚类数据
            viz_data = st.session_state.gmm_clustering_data
    
    return viz_data


def render_gmm_result(gmm_model, plot_step):
    """
    渲染 GMM 聚类结果界面
    
    参数:
        gmm_model: 训练好的 GaussianMixture 模型
        plot_step: float，网格步长
    """
    if gmm_model is None:
        return
    
    st.header("GMM 聚类结果")
    feature_names = st.session_state.get('gmm_feature_names', None)
    
    # 获取用于可视化的数据
    viz_data = get_gmm_viz_data()
    
    # 绘制聚类结果
    fig = plot_gmm_clustering(
        gmm_model,
        viz_data,
        feature_names=feature_names,
        plot_step=plot_step
    )
    st.pyplot(fig)
    
    # 计算并显示 AIC 和 BIC
    # 注意：需要使用完整的聚类数据来计算 AIC/BIC
    clustering_data = st.session_state.gmm_clustering_data
    aic, bic, log_likelihood = calculate_gmm_aic_bic(gmm_model, clustering_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("对数似然", f"{log_likelihood:.2f}")
    with col2:
        st.metric("AIC", f"{aic:.2f}")
    with col3:
        st.metric("BIC", f"{bic:.2f}")
    
    # 显示聚类信息
    st.write(f"**聚类中心（均值）:**")
    centroids_df = pd.DataFrame(
        gmm_model.means_,
        columns=st.session_state.gmm_clustering_data.columns.tolist()
    )
    st.dataframe(centroids_df)
    
    # 显示协方差信息
    st.write(f"**协方差类型:** {gmm_model.covariance_type}")
    
    # 显示混合权重
    st.write(f"**混合权重:**")
    weights_df = pd.DataFrame({
        '成分': range(1, gmm_model.n_components + 1),
        '权重': gmm_model.weights_
    })
    st.dataframe(weights_df)

