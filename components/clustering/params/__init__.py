"""
聚类参数配置组件
"""
from components.clustering.params.kmeans_params import render_kmeans_params
from components.clustering.params.gmm_params import render_gmm_params
from components.clustering.params.functional_clustering_params import render_functional_clustering_params

__all__ = [
    'render_kmeans_params',
    'render_gmm_params',
    'render_functional_clustering_params',
]

