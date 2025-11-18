"""
通用布局组件
提供可复用的页面布局组件，减少代码重复
"""
import streamlit as st
import pandas as pd
from typing import Optional, Tuple
from components.data.file_loader import load_data_file
from components.viz.plot_hist_kde import plot_hist_kde
from components.data.data_transformation import render_data_transformation
from utils.state_manager import StateManager


def render_data_upload_section(
    key_prefix: str = "",
    file_uploader_key: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    渲染数据上传部分
    
    参数:
        key_prefix: 状态键前缀，用于区分不同的聚类类型
        file_uploader_key: 文件上传器的唯一键
    
    返回:
        上传的 DataFrame 或 None
    """
    uploaded_file = st.file_uploader(
        "请上传文件",
        type=["csv", "txt", "xlsx", "xls"],
        key=file_uploader_key or f"{key_prefix}_uploader"
    )
    
    if uploaded_file is not None:
        df = load_data_file(uploaded_file, set_index=True, show_preview=True)
        if df is not None:
            state_key = f"{key_prefix}uploaded_df" if key_prefix else "uploaded_df"
            StateManager.set(state_key, df)
        return df
    return None


def render_data_preview_tabs(
    key_prefix: str = "",
    default_num: int = 15
) -> bool:
    """
    渲染数据预览标签页（数据展示和数据分布）
    
    参数:
        key_prefix: 状态键前缀
        default_num: 默认显示的变量数量
    
    返回:
        是否有数据可预览
    """
    state_key = f"{key_prefix}uploaded_df" if key_prefix else "uploaded_df"
    df = StateManager.get(state_key)
    
    if df is None:
        st.info("请先上传数据文件")
        return False
    
    tab1, tab2 = st.tabs(["数据展示", "数据分布"])
    
    with tab1:
        st.dataframe(df)
        st.write(f"**数据形状:** {df.shape[0]} 行 × {df.shape[1]} 列")
    
    with tab2:
        plot_hist_kde(
            df,
            default_num=default_num,
            button_key=f"{key_prefix}_plot_all_vars_button"
        )
    
    return True


def render_data_transformation_tabs(
    key_prefix: str = "",
    default_num: int = 15
) -> Optional[pd.DataFrame]:
    """
    渲染数据转换标签页
    
    参数:
        key_prefix: 状态键前缀
        default_num: 默认显示的变量数量
    
    返回:
        转换后的 DataFrame 或 None
    """
    uploaded_key = f"{key_prefix}uploaded_df" if key_prefix else "uploaded_df"
    transformed_key = f"{key_prefix}transformed_df" if key_prefix else "transformed_df"
    
    df = StateManager.get(uploaded_key)
    
    if df is None:
        st.info("请先在「数据预览」标签页上传数据")
        return None
    
    tab1, tab2 = st.tabs(["数据转换", "数据转换后的分布"])
    
    with tab1:
        transformation_key_prefix = f"{key_prefix}data_transformation" if key_prefix else "data_transformation"
        transformed_df = render_data_transformation(df, key_prefix=transformation_key_prefix)
        if transformed_df is not None:
            StateManager.set(transformed_key, transformed_df)
        return transformed_df
    
    with tab2:
        transformed_df = StateManager.get(transformed_key)
        if transformed_df is not None:
            plot_hist_kde(
                transformed_df,
                default_num=default_num,
                button_key=f"{key_prefix}_transformed_plot_button"
            )
        else:
            st.info("请先在「数据转换」标签页完成数据转换")
        return transformed_df


def render_clustering_workflow_tabs(
    clustering_type: str,
    key_prefix: str = "",
    additional_tabs: Optional[list] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    渲染完整的聚类工作流标签页
    
    参数:
        clustering_type: 聚类类型 ('kmeans', 'gmm', 'func')
        key_prefix: 状态键前缀
        additional_tabs: 额外的标签页配置列表，格式: [("标签名", render_function), ...]
    
    返回:
        (uploaded_df, transformed_df) 元组
    """
    # 数据上传
    uploaded_df = render_data_upload_section(
        key_prefix=key_prefix,
        file_uploader_key=f"{clustering_type}_uploader"
    )
    
    # 构建标签页列表
    tab_names = ["数据预览", "数据转换"]
    if additional_tabs:
        tab_names.extend([name for name, _ in additional_tabs])
    
    tabs = st.tabs(tab_names)
    
    # 数据预览标签页
    with tabs[0]:
        render_data_preview_tabs(key_prefix=key_prefix)
    
    # 数据转换标签页
    with tabs[1]:
        transformed_df = render_data_transformation_tabs(key_prefix=key_prefix)
    
    # 额外的标签页
    if additional_tabs:
        for idx, (tab_name, render_func) in enumerate(additional_tabs, start=2):
            with tabs[idx]:
                render_func()
    
    return uploaded_df, transformed_df

