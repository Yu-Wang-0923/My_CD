"""
统一的状态管理模块
管理 Streamlit session_state 的初始化和访问
"""
import streamlit as st
from typing import Any, Optional, Dict


class StateManager:
    """统一的状态管理器"""
    
    # 定义所有可能的状态键
    STATE_KEYS = {
        # 数据相关
        "iris_df": None,
        "uploaded_df": None,
        "transformed_df": None,
        "clustering_data": None,
        "original_clustering_data": None,
        
        # 特征相关
        "selected_features": None,
        "feature_names": None,
        
        # 模型相关
        "kmeans_model": None,
        "kmeans_history": None,
        "kmeans_params": None,
        "gmm_model": None,
        "gmm_params": None,
        "func_clustering_result": None,
        "func_params": None,
        
        # 数据转换相关
        "scaler": None,
        "is_normalized": False,
        
        # GMM 相关状态（使用前缀避免冲突）
        "gmm_uploaded_df": None,
        "gmm_transformed_df": None,
        "gmm_clustering_data": None,
        "gmm_selected_features": None,
        "gmm_feature_names": None,
        
        # Function clustering 相关状态
        "func_uploaded_df": None,
        "func_transformed_df": None,
        "func_clustering_data": None,
        "func_selected_features": None,
        "func_feature_names": None,
    }
    
    @classmethod
    def init_session_state(cls):
        """
        初始化 session_state 中的变量
        """
        for key, default_value in cls.STATE_KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        获取状态值
        
        参数:
            key: 状态键
            default: 默认值
        
        返回:
            状态值
        """
        cls.init_session_state()
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any):
        """
        设置状态值
        
        参数:
            key: 状态键
            value: 状态值
        """
        cls.init_session_state()
        st.session_state[key] = value
    
    @classmethod
    def clear(cls, key: Optional[str] = None):
        """
        清除状态值
        
        参数:
            key: 要清除的键，如果为 None 则清除所有状态
        """
        if key is None:
            # 清除所有状态
            for key in cls.STATE_KEYS:
                if key in st.session_state:
                    del st.session_state[key]
        else:
            # 清除指定键
            if key in st.session_state:
                del st.session_state[key]
    
    @classmethod
    def get_clustering_state_prefix(cls, clustering_type: str) -> str:
        """
        获取特定聚类类型的状态前缀
        
        参数:
            clustering_type: 聚类类型 ('kmeans', 'gmm', 'func')
        
        返回:
            状态前缀
        """
        prefix_map = {
            'kmeans': '',
            'gmm': 'gmm_',
            'func': 'func_',
        }
        return prefix_map.get(clustering_type, '')
    
    @classmethod
    def get_clustering_data_key(cls, clustering_type: str) -> str:
        """
        获取特定聚类类型的数据键
        
        参数:
            clustering_type: 聚类类型 ('kmeans', 'gmm', 'func')
        
        返回:
            数据键名
        """
        prefix = cls.get_clustering_state_prefix(clustering_type)
        return f"{prefix}clustering_data" if prefix else "clustering_data"


# 向后兼容的函数
def init_session_state():
    """初始化 session_state（向后兼容）"""
    StateManager.init_session_state()

