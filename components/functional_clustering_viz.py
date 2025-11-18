import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.functional_clustering import plot_functional_clustering_results

# 支持绘图中文
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def render_functional_clustering_result(result):
    """
    渲染功能聚类结果界面
    
    参数:
        result: dict，功能聚类结果
    """
    if result is None:
        return
    
    st.header("功能聚类结果")
    
    # 显示基本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("聚类数量", result['n_components'])
    with col2:
        st.metric("均值函数类型", result['mean_type'])
    with col3:
        st.metric("协方差类型", result['covariance_type'])
    
    # 绘制聚类结果
    feature_names = result.get('columns', None)
    if feature_names and len(feature_names) >= 2:
        # 如果有多列，使用前两列进行可视化
        feature_names_2d = feature_names[:2]
    else:
        feature_names_2d = None
    
    fig = plot_functional_clustering_results(result, feature_names=feature_names_2d)
    st.pyplot(fig)
    
    # 显示聚类标签
    st.subheader("聚类标签")
    labels_df = pd.DataFrame({
        '样本索引': result.get('index', range(len(result['labels']))),
        '聚类标签': result['labels'] + 1  # 从1开始编号
    })
    st.dataframe(labels_df, use_container_width=True)
    
    # 显示各聚类的样本数量
    st.subheader("各聚类的样本数量")
    unique_labels, counts = np.unique(result['labels'], return_counts=True)
    cluster_counts_df = pd.DataFrame({
        '聚类': unique_labels + 1,
        '样本数量': counts,
        '比例 (%)': (counts / len(result['labels']) * 100).round(2)
    })
    st.dataframe(cluster_counts_df, use_container_width=True)
    
    # 如果模型类型不是简化版本，显示更多信息
    if result.get('model_type') != 'simplified':
        st.subheader("模型参数")
        if 'pars_means' in result:
            st.write("**均值参数:**")
            st.dataframe(pd.DataFrame(result['pars_means']))
        
        if 'pars_covariance' in result:
            st.write("**协方差参数:**")
            st.dataframe(pd.DataFrame(result['pars_covariance']))
        
        if 'BIC' in result:
            st.metric("BIC", f"{result['BIC']:.2f}")

