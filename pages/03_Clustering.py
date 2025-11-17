import streamlit as st
import pandas as pd
import numpy as np
from components.file_loader import load_data_file
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from state import init_session_state
from services.kmeans_clustering import perform_kmeans_clustering
from services.kmeans_clustering import plot_kmeans_clustering
from services.kmeans_clustering import plot_kmeans_iteration
from services.kmeans_clustering import perform_kmeans_with_iterations

st.set_page_config(
    page_title="Clustering",
    page_icon="🔍",
    layout="wide",
)

st.title("Clustering")
st.sidebar.success("Clustering")

# 初始化 session_state
init_session_state()

df = None

uploaded_file = st.file_uploader("请上传文件", type=["csv", "txt", "xlsx", "xls"])

if uploaded_file is not None:
    df = load_data_file(uploaded_file, set_index=True, show_preview=True)
    if df is not None:
        st.session_state.uploaded_df = df

# 导入鸢尾花数据集
if st.button("导入鸢尾花数据集", key="import_iris_button"):
    iris = load_iris()
    iris_df = pd.DataFrame(
        iris.data, 
        columns=iris.feature_names
    )
    st.session_state.uploaded_df = iris_df
    st.session_state.iris_feature_names = iris.feature_names
    st.write("鸢尾花数据集:")
    st.dataframe(st.session_state.uploaded_df)

# 数据选择和准备
if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df
    
    st.header("数据准备")
    
    # 选择数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("数据集中至少需要2个数值型变量才能进行可视化聚类。")
    else:
        # 选择用于可视化的两个变量
        col1, col2 = st.columns(2)
        with col1:
            selected_feature1 = st.selectbox(
                "选择第一个变量（X轴）",
                options=numeric_cols,
                index=0,
                key="feature1_select"
            )
        with col2:
            # 确保第二个变量与第一个不同
            remaining_cols = [col for col in numeric_cols if col != selected_feature1]
            if remaining_cols:
                selected_feature2 = st.selectbox(
                    "选择第二个变量（Y轴）",
                    options=remaining_cols,
                    index=0,
                    key="feature2_select"
                )
            else:
                st.warning("没有其他数值型变量可选")
                selected_feature2 = None
        
        if selected_feature1 and selected_feature2:
            # 准备用于聚类的数据（可以选择使用所有数值型变量或仅使用选定的两个变量）
            use_all_features = st.checkbox("使用所有数值型变量进行聚类（仅用选定的两个变量可视化）", 
                                          value=False, key="use_all_features")
            
            if use_all_features:
                # 使用所有数值型变量进行聚类
                clustering_data = df[numeric_cols].copy()
            else:
                # 仅使用选定的两个变量进行聚类
                clustering_data = df[[selected_feature1, selected_feature2]].copy()
            
            # 数据标准化选项
            st.sidebar.header("数据标准化")
            normalize_data = st.sidebar.checkbox("对数据进行标准化", value=False, key="normalize_data")
            if normalize_data:
                scaler_method = st.sidebar.selectbox(
                    "标准化方法",
                    options=["StandardScaler (Z-score标准化)", 
                            "MinMaxScaler (0-1标准化)", 
                            "RobustScaler (鲁棒标准化)"],
                    index=0,
                    key="scaler_method"
                )
                
                # 根据选择创建标准化器
                if "StandardScaler" in scaler_method:
                    scaler = StandardScaler()
                elif "MinMaxScaler" in scaler_method:
                    scaler = MinMaxScaler()
                else:  # RobustScaler
                    scaler = RobustScaler()
                
                # 保存原始数据用于可视化
                st.session_state.original_clustering_data = clustering_data.copy()
                
                # 标准化数据
                clustering_data_scaled = pd.DataFrame(
                    scaler.fit_transform(clustering_data),
                    columns=clustering_data.columns,
                    index=clustering_data.index
                )
                
                # 保存标准化器
                st.session_state.scaler = scaler
                st.session_state.is_normalized = True
                
                # 可视化数据选择（标准化后还是原始数据）
                # 注意：checkbox 会自动将值保存到 st.session_state，不需要手动赋值
                st.sidebar.checkbox(
                    "使用标准化后的数据可视化（推荐）", 
                    value=True, 
                    key="use_normalized_viz",
                    help="如果启用，可视化将显示标准化后的数据，与聚类算法实际处理的数据一致"
                )
                
                # 显示标准化前后的统计信息
                with st.expander("标准化统计信息"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**标准化前:**")
                        st.dataframe(clustering_data.describe())
                    with col2:
                        st.write("**标准化后:**")
                        st.dataframe(clustering_data_scaled.describe())
                
                clustering_data = clustering_data_scaled
            else:
                st.session_state.scaler = None
                st.session_state.is_normalized = False
                st.session_state.original_clustering_data = None
                st.session_state.use_normalized_viz = False
            
            # 保存数据
            st.session_state.clustering_data = clustering_data
            
            # 保存选定的特征用于可视化
            st.session_state.selected_features = [selected_feature1, selected_feature2]
            st.session_state.feature_names = [selected_feature1, selected_feature2]
            
            # 显示选定的数据
            st.write(f"**用于聚类的数据（{len(clustering_data)}行，{len(clustering_data.columns)}列）:**")
            st.dataframe(clustering_data.head(10))
            
            # 显示用于可视化的数据（根据用户选择使用原始数据或标准化后的数据）
            if st.session_state.is_normalized and st.session_state.original_clustering_data is not None:
                if st.session_state.get('use_normalized_viz', True):
                    # 使用标准化后的数据可视化
                    viz_data = st.session_state.clustering_data[[selected_feature1, selected_feature2]]
                    st.write(f"**用于可视化的数据（标准化后，{len(viz_data)}行，2列）:**")
                else:
                    # 使用原始数据可视化
                    viz_data = st.session_state.original_clustering_data[[selected_feature1, selected_feature2]]
                    st.write(f"**用于可视化的数据（原始数据，{len(viz_data)}行，2列）:**")
            else:
                viz_data = df[[selected_feature1, selected_feature2]]
                st.write(f"**用于可视化的数据（{len(viz_data)}行，2列）:**")
            st.dataframe(viz_data.head(10))

# 执行 KMeans 聚类
if st.session_state.clustering_data is not None and st.session_state.selected_features is not None:
    st.sidebar.header("KMeans 参数设置")
    
    # 可控制的参数
    n_clusters = st.sidebar.slider("聚类数量", min_value=2, max_value=10, value=3, step=1)
    n_init = st.sidebar.selectbox("初始化次数", options=['auto', 10, 20, 50, 100], index=0)
    if n_init == 'auto':
        n_init_value = 'auto'
    else:
        n_init_value = int(n_init)
    
    random_state = st.sidebar.number_input("随机种子 (None表示随机)", 
                                          min_value=None, max_value=None, 
                                          value=None, step=1)
    if random_state is not None:
        random_state = int(random_state)
    
    # 可视化参数
    st.sidebar.header("可视化参数")
    plot_step = st.sidebar.slider("网格步长", min_value=0.01, max_value=0.1, 
                                  value=0.02, step=0.01)
    cmap_light = st.sidebar.selectbox("颜色映射", 
                                     options=['Pastel2', 'Pastel1', 'Set3', 'Set2', 'Set1'],
                                     index=0)
    
    # 迭代参数
    st.sidebar.header("迭代参数")
    max_iter = st.sidebar.slider("最大迭代次数", min_value=1, max_value=50, value=10, step=1)
    show_iteration = st.sidebar.checkbox("显示迭代过程", value=True)
    show_centroid_path = st.sidebar.checkbox("显示中心点移动路径", value=True)
    
    # 执行聚类
    if st.button("执行 KMeans 聚类", key="run_kmeans_button"):
        if show_iteration:
            # 执行带迭代历史的聚类
            kmeans, history = perform_kmeans_with_iterations(
                st.session_state.clustering_data,
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=random_state,
                return_history=True
            )
            st.session_state.kmeans_model = kmeans
            st.session_state.kmeans_history = history
        else:
            # 执行普通聚类
            kmeans = perform_kmeans_clustering(
                st.session_state.clustering_data,
                n_clusters=n_clusters,
                n_init=n_init_value,
                random_state=random_state
            )
            st.session_state.kmeans_model = kmeans
            st.session_state.kmeans_history = None
    
    # 可视化迭代过程
    if 'kmeans_model' in st.session_state and 'kmeans_history' in st.session_state:
        if st.session_state.kmeans_history is not None and show_iteration:
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
            
            # 获取用于可视化的数据（仅选定的两个特征）
            # 根据用户选择使用标准化后的数据或原始数据
            if st.session_state.is_normalized and st.session_state.original_clustering_data is not None:
                if st.session_state.get('use_normalized_viz', True):
                    # 使用标准化后的数据可视化
                    if st.session_state.clustering_data.shape[1] > 2:
                        viz_data = st.session_state.clustering_data[st.session_state.selected_features]
                    else:
                        viz_data = st.session_state.clustering_data
                else:
                    # 使用原始数据可视化
                    if st.session_state.clustering_data.shape[1] > 2:
                        viz_data = st.session_state.original_clustering_data[st.session_state.selected_features]
                    else:
                        viz_data = st.session_state.original_clustering_data
            else:
                if st.session_state.clustering_data.shape[1] > 2:
                    # 如果使用了所有特征进行聚类，需要提取选定的两个特征用于可视化
                    viz_data = st.session_state.uploaded_df[st.session_state.selected_features]
                else:
                    # 如果只使用了两个特征，直接使用聚类数据
                    viz_data = st.session_state.clustering_data
            
            # 获取上一次的中心点（用于显示路径）
            previous_centroids = None
            if iteration_to_show > 0 and show_centroid_path:
                prev_centroids = history[iteration_to_show - 1]['centroids']
                # 如果使用了所有特征，需要提取选定的两个特征的中心点
                if st.session_state.clustering_data.shape[1] > 2:
                    numeric_cols = st.session_state.clustering_data.columns.tolist()
                    idx1 = numeric_cols.index(st.session_state.selected_features[0])
                    idx2 = numeric_cols.index(st.session_state.selected_features[1])
                    previous_centroids = prev_centroids[:, [idx1, idx2]]
                    # 确保数组是 C 连续的
                    previous_centroids = np.ascontiguousarray(previous_centroids)
                else:
                    previous_centroids = np.ascontiguousarray(prev_centroids)
                
                # 如果数据被标准化了，且用户选择使用原始数据可视化，需要反标准化中心点
                if (st.session_state.is_normalized and st.session_state.scaler is not None 
                    and not st.session_state.get('use_normalized_viz', True)):
                    # 创建完整的中心点用于反标准化
                    if st.session_state.clustering_data.shape[1] > 2:
                        full_prev_centroids = prev_centroids.copy()
                        full_prev_centroids_reconstructed = np.zeros((len(full_prev_centroids), st.session_state.clustering_data.shape[1]))
                        numeric_cols = st.session_state.clustering_data.columns.tolist()
                        idx1 = numeric_cols.index(st.session_state.selected_features[0])
                        idx2 = numeric_cols.index(st.session_state.selected_features[1])
                        full_prev_centroids_reconstructed[:, idx1] = previous_centroids[:, 0]
                        full_prev_centroids_reconstructed[:, idx2] = previous_centroids[:, 1]
                        full_prev_centroids_inverse = st.session_state.scaler.inverse_transform(full_prev_centroids_reconstructed)
                        previous_centroids = full_prev_centroids_inverse[:, [idx1, idx2]]
                    else:
                        previous_centroids = st.session_state.scaler.inverse_transform(previous_centroids)
                    previous_centroids = np.ascontiguousarray(previous_centroids)
            
            # 提取当前迭代的中心点（仅选定的两个特征）
            current_centroids = current_state['centroids']
            if st.session_state.clustering_data.shape[1] > 2:
                numeric_cols = st.session_state.clustering_data.columns.tolist()
                idx1 = numeric_cols.index(st.session_state.selected_features[0])
                idx2 = numeric_cols.index(st.session_state.selected_features[1])
                current_centroids = current_centroids[:, [idx1, idx2]]
                # 确保数组是 C 连续的
                current_centroids = np.ascontiguousarray(current_centroids)
            
            # 如果数据被标准化了，且用户选择使用原始数据可视化，需要反标准化中心点
            if (st.session_state.is_normalized and st.session_state.scaler is not None 
                and not st.session_state.get('use_normalized_viz', True)):
                # 创建完整的中心点（所有特征）用于反标准化
                full_centroids = current_state['centroids'].copy()
                if st.session_state.clustering_data.shape[1] > 2:
                    # 需要重建完整的中心点
                    full_centroids_reconstructed = np.zeros((len(full_centroids), st.session_state.clustering_data.shape[1]))
                    numeric_cols = st.session_state.clustering_data.columns.tolist()
                    idx1 = numeric_cols.index(st.session_state.selected_features[0])
                    idx2 = numeric_cols.index(st.session_state.selected_features[1])
                    full_centroids_reconstructed[:, idx1] = current_centroids[:, 0]
                    full_centroids_reconstructed[:, idx2] = current_centroids[:, 1]
                    # 反标准化
                    full_centroids_inverse = st.session_state.scaler.inverse_transform(full_centroids_reconstructed)
                    # 提取选定的两个特征
                    current_centroids = full_centroids_inverse[:, [idx1, idx2]]
                else:
                    # 直接反标准化
                    current_centroids = st.session_state.scaler.inverse_transform(current_centroids)
                current_centroids = np.ascontiguousarray(current_centroids)
            
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
                    # 获取用于可视化的数据
                    if st.session_state.is_normalized and st.session_state.original_clustering_data is not None:
                        if st.session_state.get('use_normalized_viz', True):
                            # 使用标准化后的数据可视化
                            if st.session_state.clustering_data.shape[1] > 2:
                                viz_data = st.session_state.clustering_data[st.session_state.selected_features]
                            else:
                                viz_data = st.session_state.clustering_data
                        else:
                            # 使用原始数据可视化
                            if st.session_state.clustering_data.shape[1] > 2:
                                viz_data = st.session_state.original_clustering_data[st.session_state.selected_features]
                            else:
                                viz_data = st.session_state.original_clustering_data
                    else:
                        if st.session_state.clustering_data.shape[1] > 2:
                            viz_data = st.session_state.uploaded_df[st.session_state.selected_features]
                        else:
                            viz_data = st.session_state.clustering_data
                    
                    # 提取中心点（仅选定的两个特征）
                    centroids = state['centroids']
                    if st.session_state.clustering_data.shape[1] > 2:
                        numeric_cols = st.session_state.clustering_data.columns.tolist()
                        idx1 = numeric_cols.index(st.session_state.selected_features[0])
                        idx2 = numeric_cols.index(st.session_state.selected_features[1])
                        centroids = centroids[:, [idx1, idx2]]
                        # 确保数组是 C 连续的
                        centroids = np.ascontiguousarray(centroids)
                    else:
                        centroids = np.ascontiguousarray(centroids)
                    
                    # 如果数据被标准化了，且用户选择使用原始数据可视化，需要反标准化中心点
                    if (st.session_state.is_normalized and st.session_state.scaler is not None 
                        and not st.session_state.get('use_normalized_viz', True)):
                        if st.session_state.clustering_data.shape[1] > 2:
                            full_centroids_reconstructed = np.zeros((len(centroids), st.session_state.clustering_data.shape[1]))
                            numeric_cols = st.session_state.clustering_data.columns.tolist()
                            idx1 = numeric_cols.index(st.session_state.selected_features[0])
                            idx2 = numeric_cols.index(st.session_state.selected_features[1])
                            full_centroids_reconstructed[:, idx1] = centroids[:, 0]
                            full_centroids_reconstructed[:, idx2] = centroids[:, 1]
                            full_centroids_inverse = st.session_state.scaler.inverse_transform(full_centroids_reconstructed)
                            centroids = full_centroids_inverse[:, [idx1, idx2]]
                        else:
                            centroids = st.session_state.scaler.inverse_transform(centroids)
                        centroids = np.ascontiguousarray(centroids)
                    
                    previous_centroids = None
                    if i > 0 and show_centroid_path:
                        prev_centroids = history[i - 1]['centroids']
                        if st.session_state.clustering_data.shape[1] > 2:
                            numeric_cols = st.session_state.clustering_data.columns.tolist()
                            idx1 = numeric_cols.index(st.session_state.selected_features[0])
                            idx2 = numeric_cols.index(st.session_state.selected_features[1])
                            previous_centroids = prev_centroids[:, [idx1, idx2]]
                            # 确保数组是 C 连续的
                            previous_centroids = np.ascontiguousarray(previous_centroids)
                        else:
                            previous_centroids = np.ascontiguousarray(prev_centroids)
                        
                        # 如果数据被标准化了，且用户选择使用原始数据可视化，需要反标准化中心点
                        if (st.session_state.is_normalized and st.session_state.scaler is not None 
                            and not st.session_state.get('use_normalized_viz', True)):
                            if st.session_state.clustering_data.shape[1] > 2:
                                full_prev_centroids_reconstructed = np.zeros((len(previous_centroids), st.session_state.clustering_data.shape[1]))
                                numeric_cols = st.session_state.clustering_data.columns.tolist()
                                idx1 = numeric_cols.index(st.session_state.selected_features[0])
                                idx2 = numeric_cols.index(st.session_state.selected_features[1])
                                full_prev_centroids_reconstructed[:, idx1] = previous_centroids[:, 0]
                                full_prev_centroids_reconstructed[:, idx2] = previous_centroids[:, 1]
                                full_prev_centroids_inverse = st.session_state.scaler.inverse_transform(full_prev_centroids_reconstructed)
                                previous_centroids = full_prev_centroids_inverse[:, [idx1, idx2]]
                            else:
                                previous_centroids = st.session_state.scaler.inverse_transform(previous_centroids)
                            previous_centroids = np.ascontiguousarray(previous_centroids)
                    
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
    
    # 最终结果可视化
    if ('kmeans_model' in st.session_state and 
        st.session_state.kmeans_model is not None):
        st.header("最终聚类结果")
        feature_names = st.session_state.get('feature_names', None)
        
        # 获取用于可视化的数据（仅选定的两个特征）
        # 根据用户选择使用标准化后的数据或原始数据
        if st.session_state.is_normalized and st.session_state.original_clustering_data is not None:
            if st.session_state.get('use_normalized_viz', True):
                # 使用标准化后的数据可视化
                if st.session_state.clustering_data.shape[1] > 2:
                    viz_data = st.session_state.clustering_data[st.session_state.selected_features]
                else:
                    viz_data = st.session_state.clustering_data
            else:
                # 使用原始数据可视化
                if st.session_state.clustering_data.shape[1] > 2:
                    viz_data = st.session_state.original_clustering_data[st.session_state.selected_features]
                else:
                    viz_data = st.session_state.original_clustering_data
        else:
            if st.session_state.clustering_data.shape[1] > 2:
                viz_data = st.session_state.uploaded_df[st.session_state.selected_features]
            else:
                viz_data = st.session_state.clustering_data
        
        # 提取中心点（仅选定的两个特征）
        if st.session_state.clustering_data.shape[1] > 2:
            # 需要创建一个只包含选定两个特征的 KMeans 模型用于可视化
            # 提取中心点的对应维度
            numeric_cols = st.session_state.clustering_data.columns.tolist()
            idx1 = numeric_cols.index(st.session_state.selected_features[0])
            idx2 = numeric_cols.index(st.session_state.selected_features[1])
            viz_centroids = st.session_state.kmeans_model.cluster_centers_[:, [idx1, idx2]]
            # 确保数组是 C 连续的
            viz_centroids = np.ascontiguousarray(viz_centroids)
        else:
            viz_centroids = st.session_state.kmeans_model.cluster_centers_
            viz_centroids = np.ascontiguousarray(viz_centroids)
        
        # 如果数据被标准化了，且用户选择使用原始数据可视化，需要反标准化中心点
        if (st.session_state.is_normalized and st.session_state.scaler is not None 
            and not st.session_state.get('use_normalized_viz', True)):
            if st.session_state.clustering_data.shape[1] > 2:
                # 需要重建完整的中心点用于反标准化
                full_centroids_reconstructed = np.zeros((len(viz_centroids), st.session_state.clustering_data.shape[1]))
                numeric_cols = st.session_state.clustering_data.columns.tolist()
                idx1 = numeric_cols.index(st.session_state.selected_features[0])
                idx2 = numeric_cols.index(st.session_state.selected_features[1])
                full_centroids_reconstructed[:, idx1] = viz_centroids[:, 0]
                full_centroids_reconstructed[:, idx2] = viz_centroids[:, 1]
                # 反标准化
                full_centroids_inverse = st.session_state.scaler.inverse_transform(full_centroids_reconstructed)
                # 提取选定的两个特征
                viz_centroids = full_centroids_inverse[:, [idx1, idx2]]
            else:
                # 直接反标准化
                viz_centroids = st.session_state.scaler.inverse_transform(viz_centroids)
            viz_centroids = np.ascontiguousarray(viz_centroids)
        
        # 创建临时 KMeans 对象用于可视化
        from sklearn.cluster import KMeans
        viz_kmeans = KMeans(n_clusters=st.session_state.kmeans_model.n_clusters, 
                           n_init=1, max_iter=1, init=viz_centroids)
        # 调用 fit 来初始化所有内部属性（包括 _n_threads 等）
        viz_kmeans.fit(viz_data)
        # 然后设置我们提取的中心点
        viz_kmeans.cluster_centers_ = viz_centroids
        viz_kmeans.labels_ = st.session_state.kmeans_model.labels_
        
        fig = plot_kmeans_clustering(
            viz_kmeans,
            viz_data,
            feature_names=feature_names,
            plot_step=plot_step,
            cmap_light=cmap_light
        )
        st.pyplot(fig)
        
        # 显示聚类信息
        st.write(f"**聚类中心（所有特征）:**")
        # 如果数据被标准化了，显示反标准化后的中心点
        if st.session_state.is_normalized and st.session_state.scaler is not None:
            centroids_original = st.session_state.scaler.inverse_transform(
                st.session_state.kmeans_model.cluster_centers_
            )
            centroids_df = pd.DataFrame(
                centroids_original,
                columns=st.session_state.clustering_data.columns.tolist()
            )
            st.write("*（已反标准化到原始数据范围）*")
        else:
            centroids_df = pd.DataFrame(
                st.session_state.kmeans_model.cluster_centers_,
                columns=st.session_state.clustering_data.columns.tolist()
            )
        st.dataframe(centroids_df)