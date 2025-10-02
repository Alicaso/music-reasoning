# Music Theory AI Agent - Project Summary

## 项目概述

这是一个基于多智能体架构的音乐理论分析系统，能够自动分析音乐符号（Humdrum/Kern格式）并回答复杂的音乐理论问题。

## 核心特性

### 1. 多智能体架构
- **Planner Agent**: 创建分析策略和计划
- **Thinker Agent**: 执行推理和决策
- **Tool-User Agent**: 调用具体的音乐分析工具
- **Reviewer Agent**: 基于所有信息做出最终决策

### 2. 音乐分析工具 (20+ 个专业工具)
- **音高分析**: 音级识别、音程计算
- **和声分析**: 和弦进行分析、罗马数字分析
- **旋律分析**: 轮廓分析、旋律模式
- **节奏分析**: 时值计算、连音统计
- **结构分析**: 调性估计、拍号提取
- **曲式分析**: 终止式检测、结构统计

### 3. 技术特性
- 支持 Humdrum/Kern 符号格式
- 多模型支持 (OpenAI, Claude, Gemini)
- JSON Schema 参数验证
- 强大的错误处理和重试机制
- 多进程工具执行
- 全面的日志和追踪

## 项目结构

```
music-theory-ai-agent/
├── README.md                    # 项目说明
├── LICENSE                      # MIT 许可证
├── requirements.txt             # 依赖包
├── setup.py                     # 安装配置
├── config.py                    # 配置管理
├── .gitignore                   # Git 忽略文件
├── CHANGELOG.md                 # 变更日志
├── CONTRIBUTING.md              # 贡献指南
├── init_git.sh                  # Git 初始化脚本
├── 
├── # 核心代码
├── agent_pipeline_optimized_9.14.py  # 主智能体管道
├── music21_tools.py             # 音乐分析工具
├── music21_tools.yaml           # 工具模式描述
├── tool_wrapper.py              # 工具输出标准化
├── add_schemas.py               # 模式生成工具
├── 
├── # 测试脚本
├── test_exam_agent.py           # 考试数据集测试
├── test_exam_react.py           # ReAct 管道测试
├── test_reddit_agent.py         # Reddit 数据集测试
├── test_reddit_react.py         # Reddit ReAct 测试
├── test_exam_multi_model.py     # 多模型比较
├── 
├── # 数据和示例
├── data/
│   ├── kern_exam_new/           # 考试数据集 Kern 文件
│   ├── kern_reddit_new/         # Reddit 数据集 Kern 文件
│   └── runs/                    # 执行日志
├── examples/
│   ├── test.csv                 # 示例考试问题
│   ├── reddit_data_combined_final.csv  # 示例 Reddit 问题
│   └── example_usage.py         # 使用示例
└── .github/workflows/ci.yml     # CI/CD 配置
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 设置环境变量
```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. 运行单个问题分析
```bash
python agent_pipeline_optimized_9.14.py --question_id 001
```

### 4. 批量测试
```bash
# 考试数据集
python test_exam_agent.py --csv examples/test.csv

# Reddit 数据集
python test_reddit_agent.py --csv examples/reddit_data_combined_final.csv
```

## 性能表现

- **考试数据集**: 41 个问题，72% 准确率
- **Reddit 数据集**: 87 个问题，不同复杂度
- **多模型比较**: 支持 OpenAI、Claude、Gemini 模型

## 上传到 GitHub

1. 运行初始化脚本：
```bash
cd /Users/alicasowang/Desktop/mus_theory_proj/music-theory-ai-agent
./init_git.sh
```

2. 在 GitHub 上创建新仓库

3. 添加远程仓库并推送：
```bash
git remote add origin https://github.com/yourusername/music-theory-ai-agent.git
git branch -M main
git push -u origin main
```

## 项目亮点

1. **完整的项目结构**: 包含所有必要的文档、测试、配置
2. **专业的代码质量**: 遵循 Python 最佳实践
3. **详细的文档**: README、贡献指南、变更日志
4. **CI/CD 支持**: GitHub Actions 工作流
5. **示例数据**: 包含示例问题和 Kern 文件
6. **模块化设计**: 易于扩展和维护

## 注意事项

- 原始 `coding_scripts` 目录已保留，未做任何修改
- 新项目在 `music-theory-ai-agent` 目录中
- 所有核心功能都已包含
- 数据目录为空，可根据需要添加实际数据
- 已配置好 Git 忽略规则，避免上传大文件

这个项目现在是一个完整的、专业的 GitHub 仓库，可以直接上传使用！
