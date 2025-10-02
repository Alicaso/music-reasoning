#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具包装器：将music21_tools.py中的函数包装为统一的错误返回格式
"""

import functools
import re
from typing import Any, Dict, Union

def standardize_tool_output(func):
    """
    装饰器：将工具函数的输出标准化为 {"ok": bool, "data": str} 格式
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # 检查结果是否已经是标准格式
            if isinstance(result, dict) and "ok" in result and "data" in result:
                return result
            
            # 将字符串结果转换为标准格式
            result_str = str(result)
            
            # 检查是否包含错误信息
            if contains_error(result_str):
                return {"ok": False, "data": result_str}
            else:
                return {"ok": True, "data": result_str}
                
        except Exception as e:
            return {"ok": False, "data": f"执行错误: {str(e)}"}
    
    return wrapper

def contains_error(s: str) -> bool:
    """检查字符串是否包含错误信息"""
    error_patterns = [
        r"\berror\b",
        r"\bError\b", 
        r"\bERROR\b",
        r"失败",
        r"无效",
        r"不存在",
        r"not found",
        r"invalid",
        r"failed",
        r"exception"
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, s, flags=re.IGNORECASE):
            return True
    return False

# 导入并包装所有工具函数
try:
    from music21_tools import *
    
    # 获取所有工具函数并应用装饰器
    import music21_tools
    import inspect
    
    for name in dir(music21_tools):
        obj = getattr(music21_tools, name)
        if (inspect.isfunction(obj) and 
            not name.startswith("_") and 
            obj.__module__ == music21_tools.__name__):
            # 应用装饰器
            wrapped_func = standardize_tool_output(obj)
            # 标记包装后函数的归属模块，便于多进程与枚举
            try:
                wrapped_func.__module__ = __name__
            except Exception:
                pass
            globals()[name] = wrapped_func
            
except ImportError as e:
    print(f"警告: 无法导入music21_tools: {e}")
    print("请确保music21_tools.py文件存在且可导入")
