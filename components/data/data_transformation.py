import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def render_data_transformation(data, key_prefix="transformation"):
    """
    渲染数据转换界面，提供不转换和多种标准化选项
    
    参数:
        data: pandas DataFrame，要转换的数据
        key_prefix: str，用于生成唯一key的前缀
    
    返回:
        pandas DataFrame，转换后的数据
    """
    if data is None or data.empty:
        st.warning("数据为空，无法进行转换")
        return data
    
    # 数据转换选项
    transformation_method = st.selectbox(
        "数据转换方法",
        options=[
            "不转换",
            "StandardScaler (Z-score标准化)",
            "MinMaxScaler (0-1标准化)",
            "RobustScaler (鲁棒标准化)"
        ],
        index=0,
        key=f"{key_prefix}_method"
    )
    
    if transformation_method == "不转换":
        # 不转换，直接返回原数据
        st.session_state.scaler = None
        st.session_state.is_normalized = False
        st.session_state.original_clustering_data = None
        st.session_state.use_normalized_viz = False
        return data
    
    # 根据选择创建标准化器
    if "StandardScaler" in transformation_method:
        scaler = StandardScaler()
    elif "MinMaxScaler" in transformation_method:
        scaler = MinMaxScaler()
    else:  # RobustScaler
        scaler = RobustScaler()
    
    # 保存原始数据用于可视化
    st.session_state.original_clustering_data = data.copy()
    
    # 转换数据
    transformed_data = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    
    # 保存标准化器
    st.session_state.scaler = scaler
    st.session_state.is_normalized = True
    
    # 可视化数据选择（标准化后还是原始数据）
    st.checkbox(
        "使用标准化后的数据可视化（推荐）", 
        value=True, 
        key=f"{key_prefix}_use_normalized_viz",
        help="如果启用，可视化将显示标准化后的数据，与聚类算法实际处理的数据一致"
    )
    
    # 显示转换前后的统计信息
    with st.expander("转换统计信息", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**转换前:**")
            st.dataframe(data.describe())
        with col2:
            st.write("**转换后:**")
            st.dataframe(transformed_data.describe())
    
    return transformed_data

