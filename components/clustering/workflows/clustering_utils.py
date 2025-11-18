import streamlit as st
import numpy as np
import pandas as pd


def get_viz_data():
    """
    获取用于可视化的数据（仅选定的两个特征）
    聚类结果可视化总是使用标准化后的数据（如果数据被标准化了）
    
    返回:
        pandas DataFrame: 用于可视化的数据
    """
    selected_features = st.session_state.selected_features
    
    # 聚类结果可视化总是使用标准化后的数据
    if st.session_state.is_normalized and st.session_state.clustering_data is not None:
        # 使用标准化后的数据可视化
        if st.session_state.clustering_data.shape[1] > 2:
            viz_data = st.session_state.clustering_data[selected_features]
        else:
            viz_data = st.session_state.clustering_data
    else:
        # 如果数据没有被标准化，使用原始数据
        if st.session_state.clustering_data.shape[1] > 2:
            # 如果使用了所有特征进行聚类，需要提取选定的两个特征用于可视化
            if 'transformed_df' in st.session_state and st.session_state.transformed_df is not None:
                viz_data = st.session_state.transformed_df[selected_features]
            elif 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
                viz_data = st.session_state.uploaded_df[selected_features]
            else:
                viz_data = st.session_state.clustering_data[selected_features]
        else:
            # 如果只使用了两个特征，直接使用聚类数据
            viz_data = st.session_state.clustering_data
    
    return viz_data


def extract_centroids_2d(centroids_full, selected_features):
    """
    从完整中心点中提取选定的两个特征
    
    参数:
        centroids_full: numpy array，完整的中心点
        selected_features: list，选定的特征列表
    
    返回:
        numpy array，提取的中心点（2D）
    """
    if st.session_state.clustering_data.shape[1] > 2:
        numeric_cols = st.session_state.clustering_data.columns.tolist()
        idx1 = numeric_cols.index(selected_features[0])
        idx2 = numeric_cols.index(selected_features[1])
        centroids_2d = centroids_full[:, [idx1, idx2]]
    else:
        centroids_2d = centroids_full
    
    return np.ascontiguousarray(centroids_2d)


def inverse_transform_centroids(centroids_2d, selected_features):
    """
    聚类结果可视化总是使用标准化后的数据，所以不需要反标准化中心点
    此函数现在直接返回中心点，不做任何转换
    
    参数:
        centroids_2d: numpy array，2D中心点
        selected_features: list，选定的特征列表
    
    返回:
        numpy array，中心点（不做反标准化）
    """
    # 聚类结果可视化总是使用标准化后的数据，所以不需要反标准化
    # 直接返回中心点
    return centroids_2d

