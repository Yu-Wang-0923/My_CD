"""
状态管理模块（向后兼容）
推荐使用 utils.state_manager.StateManager
"""
from utils.state_manager import StateManager, init_session_state

# 导出以保持向后兼容
__all__ = ['init_session_state', 'StateManager']

