"""
基础类模块
提供模型、服务和页面的基类
"""

from core.base.model_base import BaseModel
from core.base.service_base import BaseService
from core.base.page_base import BasePage

__all__ = [
    'BaseModel',
    'BaseService',
    'BasePage',
]

