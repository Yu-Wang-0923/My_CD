import streamlit as st
import pandas as pd
import numpy as np


def render_feature_selection(df, key_prefix=""):
    """
    渲染特征选择界面
    
    参数:
        df: pandas DataFrame，输入数据
        key_prefix: str，用于区分不同标签页的 key 前缀，默认""
    
    返回:
        tuple: (selected_feature1, selected_feature2, clustering_data, use_all_features) 或 None
    """
    # 选择数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("数据集中至少需要2个数值型变量才能进行可视化聚类。")
        return None
    
    # 选择用于可视化的两个变量
    col1, col2 = st.columns(2)
    with col1:
        selected_feature1 = st.selectbox(
            "选择第一个变量（X轴）",
            options=numeric_cols,
            index=0,
            key=f"{key_prefix}_feature1_select" if key_prefix else "feature1_select"
        )
    with col2:
        # 确保第二个变量与第一个不同
        remaining_cols = [col for col in numeric_cols if col != selected_feature1]
        if remaining_cols:
            selected_feature2 = st.selectbox(
                "选择第二个变量（Y轴）",
                options=remaining_cols,
                index=0,
                key=f"{key_prefix}_feature2_select" if key_prefix else "feature2_select"
            )
        else:
            st.warning("没有其他数值型变量可选")
            selected_feature2 = None
    
    if not selected_feature1 or not selected_feature2:
        return None
    
    # 准备用于聚类的数据（可以选择使用所有数值型变量或仅使用选定的两个变量）
    use_all_features = st.checkbox("使用所有数值型变量进行聚类（仅用选定的两个变量可视化）", 
                                  value=False, key=f"{key_prefix}_use_all_features" if key_prefix else "use_all_features")
    
    if use_all_features:
        # 使用所有数值型变量进行聚类
        clustering_data = df[numeric_cols].copy()
    else:
        # 仅使用选定的两个变量进行聚类
        clustering_data = df[[selected_feature1, selected_feature2]].copy()
    
    return selected_feature1, selected_feature2, clustering_data, use_all_features


def render_data_preview(clustering_data, df, selected_feature1, selected_feature2):
    """
    渲染数据预览界面
    
    参数:
        clustering_data: pandas DataFrame，用于聚类的数据
        df: pandas DataFrame，原始数据
        selected_feature1: str，选定的第一个特征
        selected_feature2: str，选定的第二个特征
    """
    st.write(f"**用于聚类的数据（{len(clustering_data)}行，{len(clustering_data.columns)}列）:**")
    st.dataframe(clustering_data.head(10))
    
    # 显示用于可视化的数据（根据用户选择使用原始数据或标准化后的数据）
    if st.session_state.is_normalized and st.session_state.original_clustering_data is not None:
        if st.session_state.get('use_normalized_viz', True):
            # 使用标准化后的数据可视化
            viz_data = st.session_state.clustering_data[[selected_feature1, selected_feature2]]
            st.write(f"**用于可视化的数据（标准化后，{len(viz_data)}行，2列）:**")
        else:
            # 使用原始数据可视化
            viz_data = st.session_state.original_clustering_data[[selected_feature1, selected_feature2]]
            st.write(f"**用于可视化的数据（原始数据，{len(viz_data)}行，2列）:**")
    else:
        viz_data = df[[selected_feature1, selected_feature2]]
        st.write(f"**用于可视化的数据（{len(viz_data)}行，2列）:**")
    st.dataframe(viz_data.head(10))
