import streamlit as st


def init_session_state():
    """
    初始化 session_state 中的变量
    """
    if 'iris_df' not in st.session_state:
        st.session_state.iris_df = None
    if 'clustering_data' not in st.session_state:
        st.session_state.clustering_data = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'kmeans_model' not in st.session_state:
        st.session_state.kmeans_model = None
    if 'kmeans_history' not in st.session_state:
        st.session_state.kmeans_history = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'is_normalized' not in st.session_state:
        st.session_state.is_normalized = False
    if 'original_clustering_data' not in st.session_state:
        st.session_state.original_clustering_data = None

