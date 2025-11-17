import streamlit as st
import pandas as pd
from components.plot_hist_kde import plot_hist_kde
from components.file_loader import load_data_file


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
    df = load_data_file(uploaded_file, set_index=True, show_preview=True)

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


