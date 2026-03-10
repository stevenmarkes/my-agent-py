# 🤖 RAG 智能助手
基于智谱 GLM + LangChain + FAISS 构建的本地私有化 RAG 智能助手，集成多工具能力、对话记忆持久化、完善的日志/异常处理，附带 Streamlit 可视化 Web 界面，适配面试演示与实际场景使用。

## ✨ 核心亮点
1. **工程化规范**：完善的日志系统、异常捕获、敏感信息保护（`.gitignore` 规范）
2. **多工具集成**：支持 PDF 问答（RAG）、文件生成/读取、Python 脚本运行
3. **私有化部署**：本地运行，数据不泄露，支持离线 PDF 检索
4. **双端交互**：终端命令行版 + Streamlit Web 可视化版，适配不同演示场景
5. **持久化能力**：对话记忆自动保存，重启后不丢失

## 🛠️ 技术栈
| 技术/库                | 用途                          |
|-------------------------|-------------------------------|
| LangChain/LangChain-Classic | 构建 RAG 流程、工具调用链    |
| FAISS                   | 本地向量存储，PDF 文本检索    |
| 智谱 GLM API            | 大语言模型推理                |
| Streamlit               | Web 可视化界面开发            |
| Python Logging          | 日志记录与问题排查            |
| JSON 持久化             | 对话记忆、配置文件管理        |

## 🚀 快速开始
### 1. 环境依赖安装
```bash
# 安装核心依赖
pip install langchain-community langchain-classic faiss-cpu zhipuai streamlit
2. 配置智谱 API Key
复制示例配置文件：cp config.example.json config.json（Windows 用 copy config.example.json config.json）
打开 config.json，填入你的智谱 API Key（从 智谱开放平台 获取）：
json
{
  "zhipu_api_key": "你的智谱API Key"
}
3. 运行方式
方式 1：终端交互版（轻量演示）
bash
运行
python my_agent.py
方式 2：Web 可视化版（直观演示）
bash
运行
streamlit run agent_web.py
# 运行后自动打开浏览器，地址：http://localhost:8501
📌 功能演示示例
1. PDF 问答（核心 RAG 能力）
plaintext
# 终端版输入
你：基于test.pdf回答文档里的核心内容有哪些？
Agent（RAG回答）：文档核心包含...（精准检索PDF内容，无幻觉）
2. 多工具调用
plaintext
# 文件生成
你：帮我写一个简单的Python加法脚本，保存为add.py
Agent：✅ 已保存文件：add.py

# 运行脚本
你：运行add.py
Agent：✅ 运行add.py成功：
1 + 2 = 3

# 读取文件
你：查看add.py
Agent：✅ 读取add.py成功（前500字）：
def add(a, b):
    return a + b

if __name__ == "__main__":
    print(1 + 2)
📁 项目结构
plaintext
├── my_agent.py          # 核心终端版代码（RAG+多工具+日志+持久化）
├── agent_web.py         # Streamlit Web可视化版代码
├── config.example.json  # API Key配置示例（无敏感信息，可公开）
├── config.json          # 本地配置文件（含真实API Key，不上传Git）
├── test.pdf             # 测试用PDF文件（用于RAG演示）
├── .gitignore           # Git忽略配置（保护敏感/缓存文件）
└── README.md            # 项目说明文档
🎯 核心特性详解
RAG 精准问答：基于 FAISS 向量检索，解决大模型幻觉问题，仅检索 PDF 内相关内容回答
异常处理：覆盖网络异常、API Key 错误、文件不存在等场景，保证程序鲁棒性
日志系统：详细记录操作日志、错误信息，便于问题排查与演示时的流程追溯
敏感信息保护：通过 .gitignore 排除 config.json，仅上传示例文件，避免 API Key 泄露
对话记忆持久化：自动保存对话记录到 memory.json，重启程序后可继续交互
