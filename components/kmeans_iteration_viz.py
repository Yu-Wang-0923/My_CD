import streamlit as st
from services.kmeans_clustering import plot_kmeans_iteration
from components.clustering_utils import get_viz_data, extract_centroids_2d, inverse_transform_centroids


def render_iteration_visualization(show_iteration, show_centroid_path, plot_step, cmap_light):
    """
    渲染迭代过程可视化界面
    
    参数:
        show_iteration: bool，是否显示迭代过程
        show_centroid_path: bool，是否显示中心点移动路径
        plot_step: float，网格步长
        cmap_light: str，颜色映射
    """
    if 'kmeans_model' not in st.session_state or 'kmeans_history' not in st.session_state:
        return
    
    if st.session_state.kmeans_history is None or not show_iteration:
        return
    
    st.header("KMeans 迭代过程可视化")
    
    # 选择要显示的迭代
    history = st.session_state.kmeans_history
    iteration_to_show = st.slider(
        "选择迭代次数", 
        min_value=0, 
        max_value=len(history) - 1, 
        value=len(history) - 1,
        step=1,
        key="iteration_slider"
    )
    
    # 显示当前迭代
    current_state = history[iteration_to_show]
    feature_names = st.session_state.get('feature_names', None)
    selected_features = st.session_state.selected_features
    
    # 获取用于可视化的数据
    viz_data = get_viz_data()
    
    # 获取上一次的中心点（用于显示路径）
    previous_centroids = None
    if iteration_to_show > 0 and show_centroid_path:
        prev_centroids = history[iteration_to_show - 1]['centroids']
        previous_centroids = extract_centroids_2d(prev_centroids, selected_features)
        previous_centroids = inverse_transform_centroids(previous_centroids, selected_features)
    
    # 提取当前迭代的中心点（仅选定的两个特征）
    current_centroids = extract_centroids_2d(current_state['centroids'], selected_features)
    current_centroids = inverse_transform_centroids(current_centroids, selected_features)
    
    fig = plot_kmeans_iteration(
        current_centroids,
        current_state['labels'],
        viz_data,
        current_state['iteration'],
        feature_names=feature_names,
        plot_step=plot_step,
        cmap_light=cmap_light,
        show_centroid_path=show_centroid_path,
        previous_centroids=previous_centroids
    )
    st.pyplot(fig)
    
    # 显示迭代信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前迭代", current_state['iteration'] + 1)
    with col2:
        st.metric("总迭代次数", len(history))
    with col3:
        st.metric("是否收敛", "是" if iteration_to_show == len(history) - 1 else "否")
    
    # 显示所有迭代的动画（可选）
    if st.checkbox("显示所有迭代动画", key="show_all_iterations"):
        st.write("正在生成迭代动画...")
        for i, state in enumerate(history):
            viz_data = get_viz_data()
            
            # 提取中心点（仅选定的两个特征）
            centroids = extract_centroids_2d(state['centroids'], selected_features)
            centroids = inverse_transform_centroids(centroids, selected_features)
            
            previous_centroids = None
            if i > 0 and show_centroid_path:
                prev_centroids = history[i - 1]['centroids']
                previous_centroids = extract_centroids_2d(prev_centroids, selected_features)
                previous_centroids = inverse_transform_centroids(previous_centroids, selected_features)
            
            fig = plot_kmeans_iteration(
                centroids,
                state['labels'],
                viz_data,
                state['iteration'],
                feature_names=feature_names,
                plot_step=plot_step,
                cmap_light=cmap_light,
                show_centroid_path=show_centroid_path,
                previous_centroids=previous_centroids
            )
            st.pyplot(fig)

