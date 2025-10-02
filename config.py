#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for Music Theory AI Agent
"""

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # File paths
    "QUESTIONS_CSV_PATH": "examples/test.csv",
    "KERN_FOLDER": "data/kern_exam_new",
    "TOOLS_MODULE": "music21_tools",
    "TOOLS_YAML": "music21_tools.yaml",
    "RUN_OUT_DIR": "data/runs",
    
    # Model settings
    "PLANNER_MODEL": "gpt-4o-mini",
    "THINKER_MODEL": "gpt-4o-mini", 
    "TOOLER_MODEL": "gpt-4o-mini",
    "REVIEWER_MODEL": "gpt-4o-mini",
    
    # Execution settings
    "MAX_STEPS": 12,
    "TOOL_RETRY": 3,
    "TOOL_EXEC_TIMEOUT": 45,
    "OPENAI_TIMEOUT": 60,
    "OPENAI_MAX_RETRIES": 2,
    "OPENAI_TEMPERATURE": 0.0,
    
    # Multiprocessing
    "MP_START_METHOD": "spawn",
}

def get_config() -> Dict[str, Any]:
    """Get configuration with environment variable overrides."""
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    for key, default_value in DEFAULT_CONFIG.items():
        env_value = os.environ.get(key)
        if env_value is not None:
            # Type conversion based on default value type
            if isinstance(default_value, bool):
                config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(default_value, int):
                try:
                    config[key] = int(env_value)
                except ValueError:
                    config[key] = default_value
            elif isinstance(default_value, float):
                try:
                    config[key] = float(env_value)
                except ValueError:
                    config[key] = default_value
            else:
                config[key] = env_value
    
    return config

def print_config():
    """Print current configuration."""
    config = get_config()
    print("Current Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
