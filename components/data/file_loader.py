import streamlit as st
import pandas as pd


def load_data_file(uploaded_file, set_index=True, show_preview=True, preview_expander_title="数据展示"):
    """
    加载上传的文件并返回DataFrame
    
    参数:
        uploaded_file: Streamlit上传的文件对象
        set_index: bool，是否将第一列设置为索引，默认True
        show_preview: bool，是否显示数据预览，默认True
        preview_expander_title: str，预览展开器的标题，默认"数据预览"
    
    返回:
        pandas DataFrame 或 None（如果读取失败）
    """
    if uploaded_file is None:
        return None
    
    try:
        # 根据文件类型读取数据
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type in ["text/plain", "text/txt"]:
            # 尝试不同的分隔符
            try:
                df = pd.read_csv(uploaded_file, delimiter="\t")
            except:
                df = pd.read_csv(uploaded_file, delimiter=None)
        else:
            # Excel文件
            df = pd.read_excel(uploaded_file)
        
        # 设置索引
        if set_index and len(df.columns) > 0:
            df = df.set_index(df.columns[0])
        
        # 显示预览
        if show_preview:
            with st.expander(preview_expander_title):
                st.dataframe(df)
                st.write(f"**数据形状:** {df.shape[0]} 行 × {df.shape[1]} 列")
        
        return df
        
    except Exception as e:
        st.error(f"文件读取失败，请检查文件格式。错误信息：{e}")
        return None

