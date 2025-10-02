#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Music Reasoning AI Agent
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="music-reasoning-ai-agent",
    version="1.0.0",
    author="Alicaso",
    author_email="your.email@example.com",
    description="A sophisticated multi-agent system for automated music reasoning analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Alicaso/music-reasoning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "music-reasoning-agent=agent_pipeline_optimized:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords="music reasoning, ai, machine learning, symbolic music, analysis, education",
    project_urls={
        "Bug Reports": "https://github.com/Alicaso/music-reasoning/issues",
        "Source": "https://github.com/Alicaso/music-reasoning",
        "Documentation": "https://github.com/Alicaso/music-reasoning#readme",
    },
)
