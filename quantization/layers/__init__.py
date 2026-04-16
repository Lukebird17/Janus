# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
量化层模块

提供各种量化层的实现：
- HybridQuantLinear: 混合量化线性层（支持稀疏矩阵+SmoothQuant+SVD）
"""

from .hybrid_quant_linear import HybridQuantLinear

__all__ = [
    'HybridQuantLinear',
]

