import streamlit as st
import pandas as pd
from components.plot_hist_kde import plot_hist_kde


st.set_page_config(
    page_title="Data Exploration",
    page_icon="🔍",
    layout="wide",
)

st.title("Data Exploration")
st.sidebar.success("Data Exploration")

df = None

uploaded_file = st.file_uploader("请上传文件", type=["csv", "txt", "xlsx", "xls"])
if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "text/txt":
        df = pd.read_csv(uploaded_file, delimiter=None)
    else:
        df = pd.read_excel(uploaded_file)
    df = df.set_index(df.columns[0])
    with st.expander("数据预览"):   
        st.dataframe(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "数据描述", 
        "数据类型", 
        "数据缺失值",
        "数据分布",
        ])

    with tab1:
        st.dataframe(df.describe())
    with tab2:
        st.dataframe(df.dtypes)
    with tab3:
        missing_values = df.isnull().sum()
        st.write("每列的缺失值数量：")
        st.dataframe(missing_values)
    with tab4:
        plot_hist_kde(df, default_num=15, button_key="tab4_show_all_button")


