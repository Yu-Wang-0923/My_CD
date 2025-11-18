"""
聚类分析页面
提供 KMeans、GMM 和功能聚类三种聚类方法
"""
import streamlit as st
from state import init_session_state
from components.clustering.workflows.clustering_workflows import (
    render_kmeans_clustering_workflow,
    render_gmm_clustering_workflow,
    render_functional_clustering_workflow,
)
from config.settings import PAGE_CONFIG

# 页面配置
page_config = PAGE_CONFIG["clustering"]
st.set_page_config(
    page_title=page_config["page_title"],
    page_icon=page_config["page_icon"],
    layout=page_config["layout"],
)

st.title("Clustering")
st.sidebar.success("Clustering")

# 初始化 session_state
init_session_state()

# 主标签页：三种聚类方法
tab1, tab2, tab3 = st.tabs([
    "k-Means clustering",
    "GMM clustering",
    "Function clustering"
])

# KMeans 聚类
with tab1:
    render_kmeans_clustering_workflow()

# GMM 聚类
with tab2:
    render_gmm_clustering_workflow()

# 功能聚类
with tab3:
    render_functional_clustering_workflow()
