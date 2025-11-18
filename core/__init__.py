"""
核心框架模块
提供基础类、异常处理和日志系统
"""

from core.exceptions import (
    ModelError,
    DataError,
    ServiceError,
    ValidationError,
)
from core.logger import get_logger, setup_logging

__all__ = [
    'ModelError',
    'DataError',
    'ServiceError',
    'ValidationError',
    'get_logger',
    'setup_logging',
]

