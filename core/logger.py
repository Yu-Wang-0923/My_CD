"""
日志系统
提供统一的日志管理
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
try:
    from config.settings import LOG_DIR, LOG_LEVEL
except ImportError:
    # 如果 config 模块还未初始化，使用默认值
    from pathlib import Path
    LOG_DIR = Path(__file__).parent.parent / "logs"
    LOG_LEVEL = "INFO"


def setup_logging(log_level: str = None):
    """
    设置日志系统
    
    参数:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = log_level or LOG_LEVEL
    
    # 创建日志目录
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 日志文件路径
    log_file = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    参数:
        name: 日志记录器名称，默认为调用模块名
    
    返回:
        Logger 对象
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'app')
    
    return logging.getLogger(name)

