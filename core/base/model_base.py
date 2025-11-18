"""
模型基类
定义所有模型的通用接口和行为
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from core.exceptions import ModelError
from core.logger import get_logger


class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self, name: str, **kwargs):
        """
        初始化模型
        
        参数:
            name: 模型名称
            **kwargs: 模型参数
        """
        self.name = name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        """
        训练模型
        
        参数:
            X: 特征数据
            y: 目标变量（可选）
            **kwargs: 其他参数
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs):
        """
        预测
        
        参数:
            X: 特征数据
            **kwargs: 其他参数
        
        返回:
            预测结果
        """
        pass
    
    def validate_data(self, X: pd.DataFrame, require_y: bool = False, y: Optional[pd.Series] = None):
        """
        验证数据
        
        参数:
            X: 特征数据
            require_y: 是否需要目标变量
            y: 目标变量
        """
        if X is None or X.empty:
            raise ModelError("输入数据不能为空")
        
        if require_y and (y is None or y.empty):
            raise ModelError("需要目标变量但未提供")
        
        if require_y and len(X) != len(y):
            raise ModelError("特征数据和目标变量的长度不匹配")
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.params.copy()
    
    def set_params(self, **params):
        """设置模型参数"""
        self.params.update(params)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"

