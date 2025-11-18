"""
常量定义
定义应用中使用的各种常量
"""

# 模型类型
MODEL_TYPES = {
    "clustering": "聚类",
    "classification": "分类",
    "regression": "回归",
    "feature_selection": "特征选择",
}

# 聚类算法
CLUSTERING_ALGORITHMS = {
    "kmeans": {
        "name": "K-Means",
        "description": "K-Means 聚类算法",
        "icon": "🔵",
    },
    "gmm": {
        "name": "Gaussian Mixture Model",
        "description": "高斯混合模型",
        "icon": "🟢",
    },
    "functional": {
        "name": "Functional Clustering",
        "description": "功能聚类",
        "icon": "🟣",
    },
}

# 分类算法（未来扩展）
CLASSIFICATION_ALGORITHMS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "逻辑回归",
        "icon": "📊",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "随机森林",
        "icon": "🌲",
    },
    "svm": {
        "name": "Support Vector Machine",
        "description": "支持向量机",
        "icon": "⚡",
    },
}

# 回归算法（未来扩展）
REGRESSION_ALGORITHMS = {
    "linear": {
        "name": "Linear Regression",
        "description": "线性回归",
        "icon": "📈",
    },
    "ridge": {
        "name": "Ridge Regression",
        "description": "岭回归",
        "icon": "🏔️",
    },
    "lasso": {
        "name": "Lasso Regression",
        "description": "Lasso 回归",
        "icon": "🎯",
    },
}

# 特征选择方法（未来扩展）
FEATURE_SELECTION_METHODS = {
    "univariate": {
        "name": "Univariate Selection",
        "description": "单变量特征选择",
        "icon": "📊",
    },
    "recursive": {
        "name": "Recursive Feature Elimination",
        "description": "递归特征消除",
        "icon": "🔄",
    },
    "importance": {
        "name": "Feature Importance",
        "description": "特征重要性",
        "icon": "⭐",
    },
}

# 数据转换方法
DATA_TRANSFORMATION_METHODS = {
    "none": "不转换",
    "standard": "StandardScaler (Z-score标准化)",
    "minmax": "MinMaxScaler (0-1标准化)",
    "robust": "RobustScaler (鲁棒标准化)",
}

