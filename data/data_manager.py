"""
数据管理器
统一管理数据的加载、处理和存储
"""
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any
from pathlib import Path
from utils.state_manager import StateManager
from components.data.file_loader import load_data_file
from components.data.data_transformation import render_data_transformation
from core.exceptions import DataError
from core.logger import get_logger


class DataManager:
    """数据管理器"""
    
    def __init__(self, key_prefix: str = ""):
        """
        初始化数据管理器
        
        参数:
            key_prefix: 状态键前缀，用于区分不同的数据源
        """
        self.key_prefix = key_prefix
        self.logger = get_logger(self.__class__.__name__)
    
    def get_uploaded_key(self) -> str:
        """获取上传数据的键"""
        return f"{self.key_prefix}uploaded_df" if self.key_prefix else "uploaded_df"
    
    def get_transformed_key(self) -> str:
        """获取转换后数据的键"""
        return f"{self.key_prefix}transformed_df" if self.key_prefix else "transformed_df"
    
    def load_data(self, uploaded_file, set_index: bool = True, show_preview: bool = True) -> Optional[pd.DataFrame]:
        """
        加载数据文件
        
        参数:
            uploaded_file: 上传的文件对象
            set_index: 是否设置索引
            show_preview: 是否显示预览
        
        返回:
            DataFrame 或 None
        """
        if uploaded_file is None:
            return None
        
        try:
            df = load_data_file(uploaded_file, set_index=set_index, show_preview=show_preview)
            if df is not None:
                StateManager.set(self.get_uploaded_key(), df)
                self.logger.info(f"数据加载成功: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise DataError(f"数据加载失败: {str(e)}")
    
    def get_uploaded_data(self) -> Optional[pd.DataFrame]:
        """获取已上传的数据"""
        return StateManager.get(self.get_uploaded_key())
    
    def transform_data(self, df: pd.DataFrame, key_prefix: str = None) -> Optional[pd.DataFrame]:
        """
        转换数据
        
        参数:
            df: 要转换的数据
            key_prefix: 转换键前缀
        
        返回:
            转换后的 DataFrame
        """
        if df is None or df.empty:
            return None
        
        try:
            transformation_key = key_prefix or f"{self.key_prefix}data_transformation"
            transformed_df = render_data_transformation(df, key_prefix=transformation_key)
            if transformed_df is not None:
                StateManager.set(self.get_transformed_key(), transformed_df)
                self.logger.info(f"数据转换成功: {transformed_df.shape}")
            return transformed_df
        except Exception as e:
            self.logger.error(f"数据转换失败: {e}")
            raise DataError(f"数据转换失败: {str(e)}")
    
    def get_transformed_data(self) -> Optional[pd.DataFrame]:
        """获取已转换的数据"""
        return StateManager.get(self.get_transformed_key())
    
    def clear_data(self):
        """清除数据"""
        StateManager.clear(self.get_uploaded_key())
        StateManager.clear(self.get_transformed_key())
        self.logger.info("数据已清除")

