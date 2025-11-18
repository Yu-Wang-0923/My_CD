import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from services.kmeans_clustering import plot_kmeans_clustering, calculate_aic_bic
from components.clustering_utils import get_viz_data, extract_centroids_2d, inverse_transform_centroids


def extract_viz_centroids(selected_features):
    """
    提取用于可视化的中心点（仅选定的两个特征）
    
    参数:
        selected_features: list，选定的特征列表
    
    返回:
        numpy array，2D中心点
    """
    viz_centroids = extract_centroids_2d(st.session_state.kmeans_model.cluster_centers_, selected_features)
    viz_centroids = inverse_transform_centroids(viz_centroids, selected_features)
    return viz_centroids


def render_final_result(plot_step, cmap_light):
    """
    渲染最终聚类结果界面
    
    参数:
        plot_step: float，网格步长
        cmap_light: str，颜色映射
    """
    if 'kmeans_model' not in st.session_state or st.session_state.kmeans_model is None:
        return
    
    st.header("最终聚类结果")
    feature_names = st.session_state.get('feature_names', None)
    selected_features = st.session_state.selected_features
    
    # 获取用于可视化的数据
    viz_data = get_viz_data()
    
    # 提取中心点（仅选定的两个特征）
    viz_centroids = extract_viz_centroids(selected_features)
    
    # 创建临时 KMeans 对象用于可视化
    viz_kmeans = KMeans(n_clusters=st.session_state.kmeans_model.n_clusters, 
                       n_init=1, max_iter=1, init=viz_centroids)
    # 调用 fit 来初始化所有内部属性（包括 _n_threads 等）
    viz_kmeans.fit(viz_data)
    # 然后设置我们提取的中心点
    viz_kmeans.cluster_centers_ = viz_centroids
    viz_kmeans.labels_ = st.session_state.kmeans_model.labels_
    
    fig = plot_kmeans_clustering(
        viz_kmeans,
        viz_data,
        feature_names=feature_names,
        plot_step=plot_step,
        cmap_light=cmap_light
    )
    st.pyplot(fig)
    
    # 计算并显示 AIC 和 BIC
    aic, bic, sse = calculate_aic_bic(st.session_state.clustering_data, st.session_state.kmeans_model)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SSE (平方误差和)", f"{sse:.2f}")
    with col2:
        st.metric("AIC", f"{aic:.2f}")
    with col3:
        st.metric("BIC", f"{bic:.2f}")
    
    # 显示聚类信息
    st.write(f"**聚类中心（所有特征）:**")
    # 如果数据被标准化了，需要正确处理反标准化
    if st.session_state.is_normalized and st.session_state.scaler is not None:
        centroids = st.session_state.kmeans_model.cluster_centers_
        centroids_n_features = centroids.shape[1]
        
        # 获取 scaler 训练时的特征数量
        if 'transformed_df' in st.session_state and st.session_state.transformed_df is not None:
            scaler_n_features = st.session_state.transformed_df.shape[1]
            scaler_columns = st.session_state.transformed_df.columns.tolist()
        elif 'original_clustering_data' in st.session_state and st.session_state.original_clustering_data is not None:
            scaler_n_features = st.session_state.original_clustering_data.shape[1]
            scaler_columns = st.session_state.original_clustering_data.columns.tolist()
        else:
            scaler_n_features = centroids_n_features
            scaler_columns = st.session_state.clustering_data.columns.tolist()
        
        # 如果 scaler 期望的特征数与中心点的特征数不同，需要重建
        if scaler_n_features != centroids_n_features:
            # 重建完整的中心点用于反标准化
            full_centroids = np.zeros((len(centroids), scaler_n_features))
            clustering_columns = st.session_state.clustering_data.columns.tolist()
            
            # 将聚类中心点的特征映射到完整特征空间
            for i, col in enumerate(clustering_columns):
                if col in scaler_columns:
                    idx = scaler_columns.index(col)
                    full_centroids[:, idx] = centroids[:, i]
            
            centroids_original = st.session_state.scaler.inverse_transform(full_centroids)
            # 提取聚类数据对应的特征
            centroids_original = centroids_original[:, [scaler_columns.index(col) for col in clustering_columns]]
        else:
            # 特征数相同，直接反标准化
            centroids_original = st.session_state.scaler.inverse_transform(centroids)
        
        centroids_df = pd.DataFrame(
            centroids_original,
            columns=st.session_state.clustering_data.columns.tolist()
        )
        st.write("*（已反标准化到原始数据范围）*")
    else:
        centroids_df = pd.DataFrame(
            st.session_state.kmeans_model.cluster_centers_,
            columns=st.session_state.clustering_data.columns.tolist()
        )
    st.dataframe(centroids_df)

