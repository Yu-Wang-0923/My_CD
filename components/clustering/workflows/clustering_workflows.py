"""
聚类工作流组件
为不同的聚类算法提供统一的工作流界面
"""
import streamlit as st
import pandas as pd
from typing import Optional, Tuple
from utils.state_manager import StateManager
from components.data.clustering_data_prep import render_feature_selection
from components.viz.plot_hist_kde import plot_hist_kde


def render_kmeans_clustering_workflow():
    """渲染 KMeans 聚类工作流"""
    key_prefix = ""
    
    # 数据上传
    uploaded_file = st.file_uploader(
        "请上传文件",
        type=["csv", "txt", "xlsx", "xls"],
        key="kmeans_uploader"
    )
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "数据预览",
        "数据转换",
        "KMeans 聚类",
        "肘部法则分析"
    ])
    
    # 标签页1: 数据预览
    with tab1:
        tab1_1, tab1_2 = st.tabs(["数据展示", "数据分布"])
        
        with tab1_1:
            if uploaded_file is not None:
                from components.data.file_loader import load_data_file
                df = load_data_file(uploaded_file, set_index=True, show_preview=True)
                StateManager.set("uploaded_df", df)
            else:
                st.info("请先上传数据文件")
        
        with tab1_2:
            df = StateManager.get("uploaded_df")
            if df is not None:
                plot_hist_kde(df, default_num=15, button_key="tab1_1_plot_all_vars_button")
            else:
                st.info("请先上传数据文件")
    
    # 标签页2: 数据转换
    with tab2:
        tab2_1, tab2_2 = st.tabs(["数据转换", "数据转换后的分布"])
        
        with tab2_1:
            df = StateManager.get("uploaded_df")
            if df is not None:
                from components.data.data_transformation import render_data_transformation
                transformed_df = render_data_transformation(df, key_prefix="data_transformation")
                StateManager.set("transformed_df", transformed_df)
            else:
                st.info("请先在「数据预览」标签页上传数据")
        
        with tab2_2:
            transformed_df = StateManager.get("transformed_df")
            if transformed_df is not None:
                plot_hist_kde(transformed_df, default_num=15, button_key="tab1_2_plot_all_vars_button")
            else:
                st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页3: KMeans 聚类
    with tab3:
        transformed_df = StateManager.get("transformed_df")
        if transformed_df is None:
            st.info("请先在「数据转换」标签页完成数据转换")
            return
        
        # 特征选择
        st.subheader("特征选择")
        feature_result = render_feature_selection(transformed_df)
        
        if feature_result is not None:
            selected_feature1, selected_feature2, clustering_data, use_all_features = feature_result
            StateManager.set("selected_features", [selected_feature1, selected_feature2])
            StateManager.set("feature_names", [selected_feature1, selected_feature2])
            StateManager.set("clustering_data", clustering_data)
        else:
            st.warning("请先选择特征")
            return
        
        # KMeans 参数设置和执行
        clustering_data = StateManager.get("clustering_data")
        if clustering_data is not None:
            st.subheader("KMeans 参数设置")
            from components.clustering.params.kmeans_params import render_kmeans_params
            params = render_kmeans_params()
            StateManager.set("kmeans_params", params)
            
            if params is not None:
                if st.button("执行 KMeans 聚类", key="run_kmeans_button"):
                    from services.clustering.kmeans_clustering import (
                        perform_kmeans_clustering,
                        perform_kmeans_with_iterations
                    )
                    
                    if params['show_iteration']:
                        kmeans, history = perform_kmeans_with_iterations(
                            clustering_data,
                            n_clusters=params['n_clusters'],
                            max_iter=params['max_iter'],
                            random_state=params['random_state'],
                            return_history=True
                        )
                        StateManager.set("kmeans_model", kmeans)
                        StateManager.set("kmeans_history", history)
                    else:
                        kmeans = perform_kmeans_clustering(
                            clustering_data,
                            n_clusters=params['n_clusters'],
                            n_init=params['n_init'],
                            random_state=params['random_state']
                        )
                        StateManager.set("kmeans_model", kmeans)
                        StateManager.set("kmeans_history", None)
                    st.success("聚类完成！")
                
                # 可视化结果
                kmeans_model = StateManager.get("kmeans_model")
                if kmeans_model is not None:
                    st.subheader("聚类结果可视化")
                    from components.viz.kmeans_iteration_viz import render_iteration_visualization
                    from components.viz.kmeans_result_viz import render_final_result
                    
                    render_iteration_visualization(
                        params['show_iteration'],
                        params['show_centroid_path'],
                        params['plot_step'],
                        params['cmap_light']
                    )
                    render_final_result(
                        params['plot_step'],
                        params['cmap_light']
                    )
                else:
                    st.info("请点击「执行 KMeans 聚类」按钮开始聚类")
    
    # 标签页4: 肘部法则分析
    with tab4:
        clustering_data = StateManager.get("clustering_data")
        if clustering_data is not None:
            from components.clustering.analysis.elbow_analysis import render_elbow_analysis
            render_elbow_analysis()
        else:
            st.info("请先在「数据转换」标签页完成数据转换")


def render_gmm_clustering_workflow():
    """渲染 GMM 聚类工作流"""
    key_prefix = "gmm_"
    
    # 数据上传
    uploaded_file = st.file_uploader(
        "请上传文件",
        type=["csv", "txt", "xlsx", "xls"],
        key="gmm_uploader"
    )
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "数据预览",
        "数据转换",
        "GMM 聚类",
        "肘部法则分析"
    ])
    
    # 标签页1: 数据预览
    with tab1:
        tab1_1, tab1_2 = st.tabs(["数据展示", "数据分布"])
        
        with tab1_1:
            if uploaded_file is not None:
                from components.data.file_loader import load_data_file
                df = load_data_file(uploaded_file, set_index=True, show_preview=True)
                StateManager.set("gmm_uploaded_df", df)
            else:
                st.info("请先上传数据文件")
        
        with tab1_2:
            df = StateManager.get("gmm_uploaded_df")
            if df is not None:
                plot_hist_kde(df, default_num=15, button_key="tab2_1_plot_all_vars_button")
            else:
                st.info("请先上传数据文件")
    
    # 标签页2: 数据转换
    with tab2:
        tab2_1, tab2_2 = st.tabs(["数据转换", "数据转换后的分布"])
        
        with tab2_1:
            df = StateManager.get("gmm_uploaded_df")
            if df is not None:
                from components.data.data_transformation import render_data_transformation
                transformed_df = render_data_transformation(df, key_prefix="gmm_data_transformation")
                StateManager.set("gmm_transformed_df", transformed_df)
            else:
                st.info("请先在「数据预览」标签页上传数据")
        
        with tab2_2:
            transformed_df = StateManager.get("gmm_transformed_df")
            if transformed_df is not None:
                plot_hist_kde(transformed_df, default_num=15, button_key="tab2_2_plot_all_vars_button")
            else:
                st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页3: GMM 聚类
    with tab3:
        transformed_df = StateManager.get("gmm_transformed_df")
        if transformed_df is None:
            st.info("请先在「数据转换」标签页完成数据转换")
            return
        
        # 特征选择
        st.subheader("特征选择")
        feature_result = render_feature_selection(transformed_df, key_prefix="gmm")
        
        if feature_result is not None:
            selected_feature1, selected_feature2, clustering_data, use_all_features = feature_result
            StateManager.set("gmm_selected_features", [selected_feature1, selected_feature2])
            StateManager.set("gmm_feature_names", [selected_feature1, selected_feature2])
            StateManager.set("gmm_clustering_data", clustering_data)
        else:
            st.warning("请先选择特征")
            return
        
        # GMM 参数设置和执行
        clustering_data = StateManager.get("gmm_clustering_data")
        if clustering_data is not None:
            st.subheader("GMM 参数设置")
            from components.clustering.params.gmm_params import render_gmm_params
            params = render_gmm_params()
            StateManager.set("gmm_params", params)
            
            if params is not None:
                if st.button("执行 GMM 聚类", key="run_gmm_button"):
                    from services.clustering.gmm_clustering import perform_gmm_clustering
                    gmm = perform_gmm_clustering(
                        clustering_data,
                        n_components=params['n_components'],
                        covariance_type=params['covariance_type'],
                        max_iter=params['max_iter'],
                        random_state=params['random_state']
                    )
                    StateManager.set("gmm_model", gmm)
                    st.success("GMM 聚类完成！")
                
                # 可视化结果
                gmm_model = StateManager.get("gmm_model")
                if gmm_model is not None:
                    st.subheader("GMM 聚类结果可视化")
                    from components.viz.gmm_result_viz import render_gmm_result
                    plot_step = params.get('plot_step', 0.02)
                    render_gmm_result(gmm_model, plot_step)
                else:
                    st.info("请点击「执行 GMM 聚类」按钮开始聚类")
    
    # 标签页4: 肘部法则分析
    with tab4:
        clustering_data = StateManager.get("gmm_clustering_data")
        if clustering_data is not None:
            from components.clustering.analysis.gmm_elbow_analysis import render_gmm_elbow_analysis
            render_gmm_elbow_analysis()
        else:
            st.info("请先在「GMM 聚类」标签页完成特征选择")


def render_functional_clustering_workflow():
    """渲染功能聚类工作流"""
    key_prefix = "func_"
    
    # 数据上传
    uploaded_file = st.file_uploader(
        "请上传文件",
        type=["csv", "txt", "xlsx", "xls"],
        key="func_uploader"
    )
    
    tab1, tab2, tab3 = st.tabs([
        "数据预览",
        "数据转换",
        "功能聚类"
    ])
    
    # 标签页1: 数据预览
    with tab1:
        tab1_1, tab1_2 = st.tabs(["数据展示", "数据分布"])
        
        with tab1_1:
            if uploaded_file is not None:
                from components.data.file_loader import load_data_file
                df = load_data_file(uploaded_file, set_index=True, show_preview=True)
                StateManager.set("func_uploaded_df", df)
            else:
                st.info("请先上传数据文件")
        
        with tab1_2:
            df = StateManager.get("func_uploaded_df")
            if df is not None:
                plot_hist_kde(df, default_num=15, button_key="tab3_1_plot_all_vars_button")
            else:
                st.info("请先上传数据文件")
    
    # 标签页2: 数据转换
    with tab2:
        tab2_1, tab2_2 = st.tabs(["数据转换", "数据转换后的分布"])
        
        with tab2_1:
            df = StateManager.get("func_uploaded_df")
            if df is not None:
                from components.data.data_transformation import render_data_transformation
                transformed_df = render_data_transformation(df, key_prefix="func_data_transformation")
                StateManager.set("func_transformed_df", transformed_df)
            else:
                st.info("请先在「数据预览」标签页上传数据")
        
        with tab2_2:
            transformed_df = StateManager.get("func_transformed_df")
            if transformed_df is not None:
                plot_hist_kde(transformed_df, default_num=15, button_key="tab3_2_plot_all_vars_button")
            else:
                st.info("请先在「数据转换」标签页完成数据转换")
    
    # 标签页3: 功能聚类
    with tab3:
        transformed_df = StateManager.get("func_transformed_df")
        if transformed_df is None:
            st.info("请先在「数据转换」标签页完成数据转换")
            return
        
        # 功能聚类参数设置
        st.subheader("功能聚类参数设置")
        from components.clustering.params.functional_clustering_params import render_functional_clustering_params
        params = render_functional_clustering_params()
        StateManager.set("func_params", params)
        
        if params is not None:
            if st.button("执行功能聚类", key="run_func_clustering_button"):
                from services.clustering.functional_clustering import perform_functional_clustering
                with st.spinner("正在执行功能聚类，这可能需要一些时间..."):
                    try:
                        result = perform_functional_clustering(
                            transformed_df,
                            n_components=params['n_components'],
                            mean_type=params['mean_type'],
                            covariance_type=params['covariance_type'],
                            max_iter=params['max_iter'],
                            random_state=params['random_state'],
                            verbose=True,
                            times=None,
                            params=params
                        )
                        StateManager.set("func_clustering_result", result)
                        st.success("功能聚类完成！")
                    except Exception as e:
                        st.error(f"聚类过程中出现错误: {str(e)}")
                        st.exception(e)
            
            # 显示结果
            result = StateManager.get("func_clustering_result")
            if result is not None:
                from components.viz.functional_clustering_viz import render_functional_clustering_result
                render_functional_clustering_result(result)
            else:
                st.info("请点击「执行功能聚类」按钮开始聚类")

