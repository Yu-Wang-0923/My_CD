"""
字体配置模块
用于加载和配置 matplotlib 中文字体
"""
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from config.settings import SIMHEI_FONT_PATH


def load_simhei_font():
    """
    加载 SimHei 字体文件并注册到 matplotlib
    
    返回:
        FontProperties: SimHei 字体属性对象
    """
    # 将 Path 对象转换为字符串
    font_path = str(SIMHEI_FONT_PATH)
    
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件未找到: {font_path}")
    
    # 注册字体到 matplotlib
    fontManager.addfont(font_path)
    
    # 创建字体属性对象
    font_prop = FontProperties(fname=font_path)
    
    return font_prop


def configure_matplotlib_chinese():
    """
    配置 matplotlib 以支持中文显示
    自动加载 SimHei 字体并设置相关参数
    """
    try:
        # 加载字体
        load_simhei_font()
        
        # 设置 matplotlib 字体
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        return True
    except Exception as e:
        print(f"警告: 无法加载 SimHei 字体: {e}")
        print("将使用系统默认字体")
        # 如果加载失败，尝试使用系统默认的中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False


# 自动配置（在模块导入时执行）
configure_matplotlib_chinese()

