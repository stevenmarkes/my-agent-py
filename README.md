# AI Agent 智能体
基于 Python + 智谱GLM API 实现的工程化AI智能体，集成多工具调用、对话记忆持久化、完善的日志系统和异常处理，兼顾功能性与工程化规范，适配各种需求。

## 🚀 技术栈
- 核心语言：Python 3.8+
- 大模型：智谱GLM API（chatglm_std）
- 工程化：日志系统、配置管理、异常处理、文件持久化
- 工具链：文件读写、Python代码运行、对话上下文管理

## ✨ 核心特性
### 1. 工程化设计
- 📊 双端日志：终端实时输出 + 文件日志（agent.log），方便问题排查与行为追溯
- 🔒 配置隔离：敏感信息（API Key）通过 config.json 管理，避免硬编码泄露
- ⚠️ 分类异常处理：针对网络、API认证、工具调用等场景做专属异常捕获，程序鲁棒性强

### 2. 多工具集成
- 📝 文件生成：支持将AI生成的代码/文本保存为指定文件（如 qarobot.py）
- 📖 文件读取：安全读取本地文件内容（限制显示前500字，避免输出过载）
- ▶️ 代码运行：执行本地Python文件，含10秒超时保护，避免死循环阻塞

### 3. 对话记忆持久化
- 历史对话自动保存到 memory.json，重启程序不丢失上下文
- 记忆格式兼容智谱GLM API规范，无缝衔接大模型调用

## 📦 快速开始（一键运行）
### 1. 克隆仓库
git clone https://github.com/stevenmarkes/my-agent-py.git
cd my-agent-py

### 2. 安装依赖
pip install zhipuai requests

### 3. 配置 API Key
1. 复制仓库中的 config.example.json 文件，并重命名为 config.json；
2. 在 config.json 中填写你的智谱 GLM API Key：
{"zhipu_api_key": "你的智谱API Key"}

### 4. 启动智能体
python my_agent.py

## 📋 功能演示（示例指令）
运行后输入以下指令即可体验对应功能：
- 指令：帮我写一个本地问答机器人，保存为 qarobot.py → 功能：调用文件生成工具，自动创建 qarobot.py
- 指令：查看 qarobot.py → 功能：调用文件读取工具，返回文件前 500 字内容
- 指令：运行 qarobot.py → 功能：调用代码运行工具，执行文件并返回结果
- 指令：退出 → 功能：保存对话记忆并安全退出程序

## 📂 项目目录结构
my-agent-py/
├── my_agent.py          # 主程序（核心逻辑）
├── config.example.json  # 配置示例文件（可上传）
├── config.json          # 敏感配置（.gitignore忽略，不上传）
├── .gitignore           # Git忽略规则（保护敏感文件）
├── memory.json          # 对话记忆（.gitignore忽略）
├── agent.log            # 日志文件（.gitignore忽略）
└── README.md            # 项目说明文档

## 🛡️ 安全与规范
- 所有敏感文件（config.json、memory.json、agent.log）均通过 .gitignore 排除，避免 API Key 等隐私信息泄露；
- 代码运行工具添加超时保护（10 秒），避免恶意/错误代码导致程序阻塞；
- 文件读写做权限异常捕获，提升程序容错性。
