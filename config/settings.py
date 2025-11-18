"""
应用设置
统一管理应用的各种配置参数
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 资源目录
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 字体配置
SIMHEI_FONT_PATH = FONTS_DIR / "SimHei.ttf"

# 支持的文件类型
SUPPORTED_FILE_TYPES = ["csv", "txt", "xlsx", "xls"]

# 默认绘图参数
DEFAULT_PLOT_PARAMS = {
    "default_num": 15,
    "plot_step": 0.02,
    "cmap_light": "Pastel2",
}

# 日志级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 页面配置
PAGE_CONFIG = {
    "home": {
        "page_title": "EA",
        "page_icon": "☺️",
        "layout": "wide",
    },
    "data_exploration": {
        "page_title": "Data Exploration",
        "page_icon": "🔍",
        "layout": "wide",
    },
    "clustering": {
        "page_title": "Clustering",
        "page_icon": "🔍",
        "layout": "centered",
    },
    "feature_selection": {
        "page_title": "Feature Selection",
        "page_icon": "🎯",
        "layout": "wide",
    },
    "classification": {
        "page_title": "Classification",
        "page_icon": "📊",
        "layout": "wide",
    },
    "regression": {
        "page_title": "Regression",
        "page_icon": "📈",
        "layout": "wide",
    },
}

