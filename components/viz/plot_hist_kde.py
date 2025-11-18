import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 支持绘图中文 - 使用统一的字体配置
from components.utils.font_config import configure_matplotlib_chinese
configure_matplotlib_chinese()


def plot_hist_kde(df, default_num=15, button_key="plot_hist_kde_button"):
    """
    绘制数值型变量的直方图和KDE密度图
    
    参数:
        df: pandas DataFrame，要绘制的数据框
        default_num: int，默认显示的变量数量，默认15
        button_key: str，Streamlit按钮的唯一key
    
    返回:
        None（直接使用st.pyplot显示图表）
    """
    if df is None:
        st.write("数据框为空，无法绘图")
        return
    
    # 找出演示用的数值型变量
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 默认画前default_num个
    st.write(f"默认画前{default_num}个数值型变量")
    show_all = st.button("画所有变量", key=button_key)
    
    if show_all:
        plot_cols = numeric_cols
    else:
        plot_cols = numeric_cols[:default_num] if len(numeric_cols) >= default_num else numeric_cols[:]
    
    num_plots = len(plot_cols)
    
    if num_plots > 0:
        cols = 3
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        ax_label_fontsize = 20  # 设置坐标轴字体大小
        tick_fontsize = 12      # 设置刻度字体大小

        # 绘制每个选择的变量
        for idx, col in enumerate(plot_cols):
            sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
            axes[idx].set_xlabel(col, fontsize=ax_label_fontsize)
            axes[idx].set_ylabel('Count', fontsize=ax_label_fontsize)
            axes[idx].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # 多余的子图清空
        for extra_idx in range(len(plot_cols), len(axes)):
            axes[extra_idx].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("没有数值型变量可供绘图")

