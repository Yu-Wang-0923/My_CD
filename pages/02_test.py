import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="检查数据缺失值",
    page_icon="🔍",
    layout="wide",
    # layout="centered",
)

st.write("请上传文件")
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx", "xls"])

df = None

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "text/txt":
        df = pd.read_csv(uploaded_file, delimiter=None)
    else:
        df = pd.read_excel(uploaded_file)
    df = df.set_index(df.columns[0])
    st.write("Preview of the data:")

    with st.expander("查看数据帧"):
        st.dataframe(df)    
        st.dataframe(df.isnull())

    
