import streamlit as st
import pandas as pd
from components.file_loader import load_data_file
from state import init_session_state
from services.kmeans_clustering import perform_kmeans_clustering
from services.kmeans_clustering import perform_kmeans_with_iterations
from components.clustering_data_prep import render_feature_selection, render_data_preview
from components.data_transformation import render_data_transformation
from components.kmeans_params import render_kmeans_params
from components.elbow_analysis import render_elbow_analysis
from components.kmeans_iteration_viz import render_iteration_visualization
from components.kmeans_result_viz import render_final_result
from components.plot_hist_kde import plot_hist_kde
from services.gmm_clustering import perform_gmm_clustering
from components.gmm_params import render_gmm_params
from components.gmm_result_viz import render_gmm_result
from components.gmm_elbow_analysis import render_gmm_elbow_analysis
from services.functional_clustering import perform_functional_clustering
from components.functional_clustering_params import render_functional_clustering_params
from components.functional_clustering_viz import render_functional_clustering_result

st.set_page_config(
    page_title="Clustering",
    page_icon="🔍",
    layout="centered",
)

st.title("Clustering")
st.sidebar.success("Clustering")

# 初始化 session_state
init_session_state()

tab1, tab2, tab3 = st.tabs(["k-Means clustering", "GMM clustering", "Function clustering"])

# KMeans clustering
with tab1:
    # 数据上传
    uploaded_file = st.file_uploader("请上传文件", type=["csv", "txt", "xlsx", "xls"])
    
    tab1_1, tab1_2, tab1_3, tab1_4 = st.tabs(["数据预览", "数据转换", "KMeans 聚类", "肘部法则分析"])
    
    # 标签页1: 数据预览
    with tab1_1:
        tab1_1_1, tab1_1_2 = st.tabs(["数据展示", "数据分布"])
        
        with tab1_1_1:
            if uploaded_file is not None:
                df = load_data_file(uploaded_file, set_index=True, show_preview=True)
                st.session_state.uploaded_df = df
            else:
                st.info("请先上传数据文件")
        
        with tab1_1_2:
            if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
                df = st.session_state.uploaded_df
                plot_hist_kde(df, default_num=15, button_key="tab1_1_plot_all_vars_button")
            else:
                st.info("请先上传数据文件")
    
    # 标签页2: 数据转换
    with tab1_2:
        tab1_2_1, tab1_2_2 = st.tabs(["数据转换", "数据转换后的分布"])
        
        with tab1_2_1:
            if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
                df = st.session_state.uploaded_df
                # 对原始数据进行转换
                transformed_df = render_data_transformation(df, key_prefix="data_transformation")
                st.session_state.transformed_df = transformed_df
            else:
                st.info("请先在「数据预览」标签页上传数据")
        
        with tab1_2_2:
            if 'transformed_df' in st.session_state and st.session_state.transformed_df is not None:
                plot_hist_kde(st.session_state.transformed_df, default_num=15, button_key="tab1_2_plot_all_vars_button")
            else:
                st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页3: KMeans 聚类（包含特征选择、参数设置和结果可视化）
    with tab1_3:
        if 'transformed_df' in st.session_state and st.session_state.transformed_df is not None:
            df = st.session_state.transformed_df
            
            # 特征选择
            st.subheader("特征选择")
            feature_result = render_feature_selection(df)
            
            if feature_result is not None:
                selected_feature1, selected_feature2, clustering_data, use_all_features = feature_result
                st.session_state.selected_features = [selected_feature1, selected_feature2]
                st.session_state.feature_names = [selected_feature1, selected_feature2]
                # 保存用于聚类的数据（已经是转换后的数据）
                st.session_state.clustering_data = clustering_data
            else:
                st.warning("请先选择特征")
            
            # KMeans 参数设置
            if 'clustering_data' in st.session_state and st.session_state.clustering_data is not None:
                st.subheader("KMeans 参数设置")
                params = render_kmeans_params()
                st.session_state.kmeans_params = params
                
                # 执行聚类
                if params is not None:
                    if st.button("执行 KMeans 聚类", key="run_kmeans_button"):
                        if params['show_iteration']:
                            kmeans, history = perform_kmeans_with_iterations(
                                st.session_state.clustering_data,
                                n_clusters=params['n_clusters'],
                                max_iter=params['max_iter'],
                                random_state=params['random_state'],
                                return_history=True
                            )
                            st.session_state.kmeans_model = kmeans
                            st.session_state.kmeans_history = history
                        else:
                            kmeans = perform_kmeans_clustering(
                                st.session_state.clustering_data,
                                n_clusters=params['n_clusters'],
                                n_init=params['n_init'],
                                random_state=params['random_state']
                            )
                            st.session_state.kmeans_model = kmeans
                            st.session_state.kmeans_history = None
                        st.success("聚类完成！")
                    
                    # 可视化迭代过程
                    if 'kmeans_model' in st.session_state and st.session_state.kmeans_model is not None:
                        st.subheader("聚类结果可视化")
                        render_iteration_visualization(
                            params['show_iteration'],
                            params['show_centroid_path'],
                            params['plot_step'],
                            params['cmap_light']
                        )
                        
                        # 最终结果可视化
                        render_final_result(
                            params['plot_step'],
                            params['cmap_light']
                        )
                    else:
                        st.info("请点击「执行 KMeans 聚类」按钮开始聚类")
            else:
                st.info("请先完成特征选择")
        else:
            st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页4: 肘部法则分析
    with tab1_4:
        if 'clustering_data' in st.session_state and st.session_state.clustering_data is not None:
            render_elbow_analysis()
        else:
            st.info("请先在「数据转换」标签页完成数据转换")

with tab2:
    # GMM 聚类标签页（独立于 tab1）
    # 数据上传
    uploaded_file_gmm = st.file_uploader("请上传文件", type=["csv", "txt", "xlsx", "xls"], key="gmm_uploader")
    
    tab2_1, tab2_2, tab2_3, tab2_4 = st.tabs(["数据预览", "数据转换", "GMM 聚类", "肘部法则分析"])
    
    # 标签页1: 数据预览
    with tab2_1:
        tab2_1_1, tab2_1_2 = st.tabs(["数据展示", "数据分布"])
        
        with tab2_1_1:
            if uploaded_file_gmm is not None:
                df = load_data_file(uploaded_file_gmm, set_index=True, show_preview=True)
                st.session_state.gmm_uploaded_df = df
            else:
                st.info("请先上传数据文件")
        
        with tab2_1_2:
            if 'gmm_uploaded_df' in st.session_state and st.session_state.gmm_uploaded_df is not None:
                df = st.session_state.gmm_uploaded_df
                plot_hist_kde(df, default_num=15, button_key="tab2_1_plot_all_vars_button")
            else:
                st.info("请先上传数据文件")
    
    # 标签页2: 数据转换
    with tab2_2:
        tab2_2_1, tab2_2_2 = st.tabs(["数据转换", "数据转换后的分布"])
        
        with tab2_2_1:
            if 'gmm_uploaded_df' in st.session_state and st.session_state.gmm_uploaded_df is not None:
                df = st.session_state.gmm_uploaded_df
                # 对原始数据进行转换
                transformed_df = render_data_transformation(df, key_prefix="gmm_data_transformation")
                st.session_state.gmm_transformed_df = transformed_df
            else:
                st.info("请先在「数据预览」标签页上传数据")
        
        with tab2_2_2:
            if 'gmm_transformed_df' in st.session_state and st.session_state.gmm_transformed_df is not None:
                plot_hist_kde(st.session_state.gmm_transformed_df, default_num=15, button_key="tab2_2_plot_all_vars_button")
            else:
                st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页3: GMM 聚类（包含特征选择、参数设置和结果可视化）
    with tab2_3:
        if 'gmm_transformed_df' in st.session_state and st.session_state.gmm_transformed_df is not None:
            df = st.session_state.gmm_transformed_df
            
            # 特征选择
            st.subheader("特征选择")
            feature_result = render_feature_selection(df, key_prefix="gmm")
            
            if feature_result is not None:
                selected_feature1, selected_feature2, clustering_data, use_all_features = feature_result
                st.session_state.gmm_selected_features = [selected_feature1, selected_feature2]
                st.session_state.gmm_feature_names = [selected_feature1, selected_feature2]
                # 保存用于聚类的数据（已经是转换后的数据）
                st.session_state.gmm_clustering_data = clustering_data
            else:
                st.warning("请先选择特征")
            
            # GMM 参数设置
            if 'gmm_clustering_data' in st.session_state and st.session_state.gmm_clustering_data is not None:
                st.subheader("GMM 参数设置")
                params = render_gmm_params()
                st.session_state.gmm_params = params
                
                # 执行聚类
                if params is not None:
                    if st.button("执行 GMM 聚类", key="run_gmm_button"):
                        gmm = perform_gmm_clustering(
                            st.session_state.gmm_clustering_data,
                            n_components=params['n_components'],
                            covariance_type=params['covariance_type'],
                            max_iter=params['max_iter'],
                            random_state=params['random_state']
                        )
                        st.session_state.gmm_model = gmm
                        st.success("GMM 聚类完成！")
                    
                    # 可视化结果
                    if 'gmm_model' in st.session_state and st.session_state.gmm_model is not None:
                        st.subheader("GMM 聚类结果可视化")
                        # 渲染结果（使用独立的 GMM 数据）
                        plot_step = params.get('plot_step', 0.02)
                        render_gmm_result(st.session_state.gmm_model, plot_step)
                    else:
                        st.info("请点击「执行 GMM 聚类」按钮开始聚类")
            else:
                st.info("请先完成特征选择")
        else:
            st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页4: 肘部法则分析
    with tab2_4:
        if 'gmm_clustering_data' in st.session_state and st.session_state.gmm_clustering_data is not None:
            render_gmm_elbow_analysis()
        else:
            st.info("请先在「GMM 聚类」标签页完成特征选择")

# Function clustering
with tab3:
    # 功能聚类标签页（独立于其他标签页）
    # 数据上传
    uploaded_file_func = st.file_uploader("请上传文件", type=["csv", "txt", "xlsx", "xls"], key="func_uploader")
    
    tab3_1, tab3_2, tab3_3 = st.tabs(["数据预览", "数据转换", "功能聚类"])
    
    # 标签页1: 数据预览
    with tab3_1:
        tab3_1_1, tab3_1_2 = st.tabs(["数据展示", "数据分布"])
        
        with tab3_1_1:
            if uploaded_file_func is not None:
                df = load_data_file(uploaded_file_func, set_index=True, show_preview=True)
                st.session_state.func_uploaded_df = df
            else:
                st.info("请先上传数据文件")
        
        with tab3_1_2:
            if 'func_uploaded_df' in st.session_state and st.session_state.func_uploaded_df is not None:
                df = st.session_state.func_uploaded_df
                plot_hist_kde(df, default_num=15, button_key="tab3_1_plot_all_vars_button")
            else:
                st.info("请先上传数据文件")
    
    # 标签页2: 数据转换
    with tab3_2:
        tab3_2_1, tab3_2_2 = st.tabs(["数据转换", "数据转换后的分布"])
        
        with tab3_2_1:
            if 'func_uploaded_df' in st.session_state and st.session_state.func_uploaded_df is not None:
                df = st.session_state.func_uploaded_df
                # 对原始数据进行转换
                transformed_df = render_data_transformation(df, key_prefix="func_data_transformation")
                st.session_state.func_transformed_df = transformed_df
            else:
                st.info("请先在「数据预览」标签页上传数据")
        
        with tab3_2_2:
            if 'func_transformed_df' in st.session_state and st.session_state.func_transformed_df is not None:
                plot_hist_kde(st.session_state.func_transformed_df, default_num=15, button_key="tab3_2_plot_all_vars_button")
            else:
                st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页3: 功能聚类
    with tab3_3:
        if 'func_transformed_df' in st.session_state and st.session_state.func_transformed_df is not None:
            df = st.session_state.func_transformed_df
            
            # 功能聚类参数设置
            st.subheader("功能聚类参数设置")
            params = render_functional_clustering_params()
            st.session_state.func_params = params
            
            # 执行聚类
            if params is not None:
                if st.button("执行功能聚类", key="run_func_clustering_button"):
                    with st.spinner("正在执行功能聚类，这可能需要一些时间..."):
                        try:
                            # 将参数传递给函数
                            result = perform_functional_clustering(
                                df,
                                n_components=params['n_components'],
                                mean_type=params['mean_type'],
                                covariance_type=params['covariance_type'],
                                max_iter=params['max_iter'],
                                random_state=params['random_state'],
                                verbose=True,
                                times=None,  # 可以后续添加时间点选择功能
                                params=params  # 传递完整参数字典
                            )
                            st.session_state.func_clustering_result = result
                            st.success("功能聚类完成！")
                        except Exception as e:
                            st.error(f"聚类过程中出现错误: {str(e)}")
                            st.exception(e)
                
                # 显示结果
                if 'func_clustering_result' in st.session_state and st.session_state.func_clustering_result is not None:
                    render_functional_clustering_result(st.session_state.func_clustering_result)
                else:
                    st.info("请点击「执行功能聚类」按钮开始聚类")
        else:
            st.info("请先在「数据转换」标签页完成数据转换")
