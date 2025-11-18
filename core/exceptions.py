"""
自定义异常类
提供统一的异常处理机制
"""


class BaseAppError(Exception):
    """应用基础异常类"""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ModelError(BaseAppError):
    """模型相关错误"""
    pass


class DataError(BaseAppError):
    """数据相关错误"""
    pass


class ServiceError(BaseAppError):
    """服务相关错误"""
    pass


class ValidationError(BaseAppError):
    """验证相关错误"""
    pass


class ConfigurationError(BaseAppError):
    """配置相关错误"""
    pass

