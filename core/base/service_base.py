"""
服务基类
定义所有服务的通用接口和行为
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from core.exceptions import ServiceError
from core.logger import get_logger


class BaseService(ABC):
    """服务基类"""
    
    def __init__(self, name: str):
        """
        初始化服务
        
        参数:
            name: 服务名称
        """
        self.name = name
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        执行服务
        
        参数:
            data: 输入数据
            **kwargs: 其他参数
        
        返回:
            服务执行结果
        """
        pass
    
    def validate_input(self, data: pd.DataFrame, **kwargs):
        """
        验证输入
        
        参数:
            data: 输入数据
            **kwargs: 其他参数
        """
        if data is None or data.empty:
            raise ServiceError("输入数据不能为空")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

