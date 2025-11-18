"""
页面基类
定义 Streamlit 页面的通用结构和行为
"""
import streamlit as st
from typing import Dict, Optional
try:
    from config.settings import PAGE_CONFIG
except ImportError:
    PAGE_CONFIG = {}
from core.logger import get_logger


class BasePage:
    """Streamlit 页面基类"""
    
    def __init__(self, page_key: str, title: str = None):
        """
        初始化页面
        
        参数:
            page_key: 页面配置键
            title: 页面标题（可选，默认使用配置）
        """
        self.page_key = page_key
        self.config = PAGE_CONFIG.get(page_key, {})
        self.title = title or self.config.get("page_title", "Page")
        self.logger = get_logger(self.__class__.__name__)
        
        # 设置页面配置
        st.set_page_config(
            page_title=self.config.get("page_title", self.title),
            page_icon=self.config.get("page_icon", "📄"),
            layout=self.config.get("layout", "wide"),
        )
    
    def render_header(self):
        """渲染页面头部"""
        st.title(self.title)
        st.sidebar.success(self.title)
    
    def render(self):
        """
        渲染页面内容
        子类需要重写此方法
        """
        self.render_header()
        st.write("页面内容待实现")
    
    def render_error(self, error: Exception):
        """
        渲染错误信息
        
        参数:
            error: 异常对象
        """
        st.error(f"发生错误: {str(error)}")
        self.logger.error(f"页面错误: {error}", exc_info=True)
    
    def render_success(self, message: str):
        """
        渲染成功信息
        
        参数:
            message: 成功消息
        """
        st.success(message)
        self.logger.info(f"页面操作成功: {message}")

