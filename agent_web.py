import streamlit as st
import json
import logging
import os
import subprocess
import requests
import sys
from zhipuai import ZhipuAI
import tempfile

# ========== 复用你原代码的所有导入 ==========
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI

# ====================== 先义所有核心函数（ ======================
# 2.1 配置读取
def load_config():
    """加载配置文件，避免API Key硬编码"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error("配置文件 config.json 不存在，已自动创建空配置")
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"zhipu_api_key": ""}, f, ensure_ascii=False, indent=2)
        return {"zhipu_api_key": ""}

# 2.2 对话记忆
def save_memory(memory):
    """保存对话记忆到文件，重启不丢失"""
    try:
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        logger.info("对话记忆已保存到 memory.json")
        return True
    except Exception as e:
        logger.error(f"保存记忆失败：{str(e)}")
        return False

def load_memory():
    """启动时加载历史对话记忆"""
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            memory = json.load(f)
        logger.info(f"成功加载{len(memory)}条历史对话记忆")
        return memory
    except FileNotFoundError:
        logger.info("无历史记忆文件，初始化空记忆")
        return []
    except json.JSONDecodeError:
        logger.error("memory.json格式错误，重置为空记忆")
        return []

# 2.3 工具函数（适配Web输出）
def write_file(filename: str, content: str):
    """生成指定文件（核心工具）"""
    invalid_chars = ["../", "/", "\\", "..\\"]
    for char in invalid_chars:
        filename = filename.replace(char, "")
    if not filename:
        return "❌ 文件名无效：请输入合法文件名（如 qarobot.py）"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"成功生成文件：{filename}")
        return f"✅ 已自动保存文件：{filename}\n可在项目目录中查看/编辑该文件"
    except PermissionError:
        logger.error(f"无权限写入 {filename}")
        return f"❌ 保存失败：没有文件写入权限，请检查目录权限"
    except Exception as e:
        logger.error(f"生成 {filename} 失败：{str(e)}")
        return f"❌ 保存 {filename} 失败：{str(e)}"

def read_file(filename: str):
    """读取本地文件内容（扩展工具）"""
    invalid_chars = ["../", "/", "\\", "..\\"]
    for char in invalid_chars:
        filename = filename.replace(char, "")
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"成功读取文件：{filename}")
        return f"✅ 读取 {filename} 成功（前 500 字）：\n{content[:500]}..."
    except FileNotFoundError:
        logger.error(f"文件 {filename} 不存在")
        return f"❌ 读取失败：{filename} 不存在\n请检查项目目录是否有该文件"
    except Exception as e:
        logger.error(f"读取 {filename} 失败：{str(e)}")
        return f"❌ 读取 {filename} 失败：{str(e)}"

def run_python_code(filename: str):
    """运行本地Python文件（扩展工具）"""
    invalid_chars = ["../", "/", "\\", "..\\"]
    for char in invalid_chars:
        filename = filename.replace(char, "")
    if not filename.endswith(".py"):
        return "❌ 仅支持运行 .py 后缀的 Python 文件"
    
    python_cmd = "python3" if sys.platform != "win32" else "python"
    try:
        result = subprocess.run(
            [python_cmd, filename],
            capture_output=True,
            encoding="utf-8",
            timeout=15
        )
        if result.returncode == 0:
            logger.info(f"成功运行 {filename}")
            return f"✅ 运行 {filename} 成功：\n{result.stdout}"
        else:
            logger.error(f"运行 {filename} 失败：{result.stderr}")
            return f"❌ 运行 {filename} 失败：\n{result.stderr}\n💡 如有缺少依赖，请先使用 pip install 安装"
    except subprocess.TimeoutExpired:
        logger.error(f"运行 {filename} 超时（15 秒）")
        return f"❌ 运行 {filename} 超时：代码运行超过 15 秒，请简化逻辑"
    except FileNotFoundError:
        logger.error(f"运行失败：{filename} 文件不存在")
        return f"❌ 运行失败：{filename} 文件不存在"
    except Exception as e:
        logger.error(f"运行 {filename} 异常：{str(e)}")
        return f"❌ 运行 {filename} 异常：{str(e)}"

# 2.4 RAG核心逻辑（适配Web上传的PDF）
def init_rag_chain_web(zhipu_api_key, pdf_file):
    """适配Web的RAG初始化（处理上传的PDF文件）"""
    # 保存上传的PDF为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_pdf_path = temp_file.name
    st.session_state.temp_pdf_path = temp_pdf_path  # 保存临时路径
    
    try:
        # 1. 初始化智谱大模型
        llm = ChatZhipuAI(api_key=zhipu_api_key, model="glm-4")
        # 2. 加载PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        logger.info(f"成功加载PDF文件，共{len(documents)}页")
        st.success(f"✅ PDF加载成功！共{len(documents)}页")
        # 3. 文档切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        # 4. 构建向量库
        embeddings = FakeEmbeddings(size=1024)
        vector_db = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        # 5. 构建RAG链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        logger.info("✅ RAG问答链初始化成功")
        return qa_chain
    except Exception as e:
        logger.error(f"❌ RAG初始化失败：{str(e)}")
        st.error(f"❌ RAG初始化失败：{str(e)}")
        return None

# ====================== 1. Web页面基础配置 ======================
# 页面样式配置
st.set_page_config(
    page_title="RAG智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_web.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化Streamlit会话状态（
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None  # RAG问答链
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 聊天记录
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()  # 复用你的记忆逻辑
if "api_key" not in st.session_state:
    st.session_state.api_key = load_config().get("zhipu_api_key", "")  # 复用你的配置读取
if "temp_pdf_path" not in st.session_state:
    st.session_state.temp_pdf_path = None  # 上传的PDF临时路径

# ====================== 3. Web界面布局 ======================
# 3.1 侧边栏：配置中心（API Key + PDF上传）
with st.sidebar:
    st.title("⚙️ 配置中心")
    
    # API Key配置
    api_key = st.text_input(
        "智谱API Key",
        value=st.session_state.api_key,
        type="password",
        help="前往 https://open.bigmodel.cn/ 获取API Key"
    )
    st.session_state.api_key = api_key
    
    # 保存API Key
    if st.button("💾 保存API Key", type="primary"):
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"zhipu_api_key": api_key}, f, ensure_ascii=False, indent=2)
        st.success("✅ API Key已保存到config.json！")
    
    st.divider()
    
    # PDF上传
    st.subheader("📄 上传PDF文档")
    pdf_file = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        help="上传后自动初始化RAG问答链"
    )
    
    # 初始化RAG链
    if st.button("🚀 初始化问答链", type="primary", disabled=not (api_key and pdf_file)):
        with st.spinner("正在初始化RAG问答链..."):
            st.session_state.rag_chain = init_rag_chain_web(api_key, pdf_file)
    
    st.divider()
    
    # 清空记忆
    if st.button("🗑️ 清空对话记忆"):
        st.session_state.memory = []
        save_memory([])
        st.session_state.chat_history = []
        st.success("✅ 对话记忆已清空！")

# 3.2 主界面：聊天交互区
st.title("🤖 RAG智能问答助手")
st.caption("核心能力：本地PDF问答 + 文件生成/读取/运行 + 对话记忆 | 基于智谱GLM + LangChain")

# 显示聊天记录
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3.3 提问输入框
if prompt := st.chat_input("请输入你的问题（示例：文档里讲了什么？/ 帮我写个Python脚本保存为test.py）"):
    # 前置校验
    if not st.session_state.api_key:
        st.error("❌ 请先在侧边栏输入并保存智谱API Key！")
    else:
        # 显示用户问题
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.memory.append({"role": "user", "content": prompt})  # 同步到记忆
        
        # 3.4 逻辑分支：RAG问答 / 工具调用 / 普通问答
        # 分支1：RAG问答（包含PDF/文档关键词）
        if st.session_state.rag_chain and ("pdf" in prompt.lower() or "文档" in prompt.lower()):
            with st.spinner("正在检索PDF文档并生成回答..."):
                try:
                    result = st.session_state.rag_chain.invoke({"query": prompt})
                    answer = result["result"]
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    # 保存记录
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.memory.append({"role": "assistant", "content": answer})
                    save_memory(st.session_state.memory)
                    logger.info(f"✅ RAG问答成功：{prompt[:20]}...")
                except Exception as e:
                    error_msg = f"❌ RAG问答失败：{str(e)}"
                    st.chat_message("assistant").markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    logger.error(error_msg)
        
        # 分支2：工具调用（包含保存/读取/运行关键词）
        elif any(keyword in prompt.lower() for keyword in ["保存为", "生成", "读取", "运行"]):
            with st.spinner("正在执行工具操作..."):
                try:
                    # 调用智谱API获取工具调用指令
                    client = ZhipuAI(api_key=st.session_state.api_key)
                    response = client.chat.completions.create(
                        model="chatglm_std",
                        messages=st.session_state.memory,
                        tools=[
                            {
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "description": "当用户说“保存为xx文件”“生成xx.py”时调用，自动生成指定文件",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {"type": "string", "description": "要保存的文件名（如qarobot.py）"},
                                            "content": {"type": "string", "description": "文件的完整代码/文本内容"}
                                        },
                                        "required": ["filename", "content"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "description": "当用户说“查看xx文件”“读取xx.py”时调用，返回文件内容",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {"type": "string", "description": "要读取的文件名"}
                                        },
                                        "required": ["filename"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "run_python_code",
                                    "description": "当用户说“运行xx.py”“执行xx文件”时调用，运行指定Python文件",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {"type": "string", "description": "要运行的Python文件名（带.py）"}
                                        },
                                        "required": ["filename"],
                                        "additionalProperties": False
                                    }
                                }
                            }
                        ],
                        tool_choice="auto",
                        temperature=0.1
                    )
                    
                    # 处理工具调用结果
                    ai_msg = response.choices[0].message
                    st.session_state.memory.append({"role": "assistant", "content": ai_msg.content})
                    
                    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                        for tc in ai_msg.tool_calls:
                            func_name = tc.function.name
                            args = json.loads(tc.function.arguments)
                            if func_name == "write_file":
                                res = write_file(args.get("filename", ""), args.get("content", ""))
                            elif func_name == "read_file":
                                res = read_file(args.get("filename", ""))
                            elif func_name == "run_python_code":
                                res = run_python_code(args.get("filename", ""))
                            else:
                                res = f"❌ 不支持的工具：{func_name}"
                            
                            # 显示工具调用结果
                            with st.chat_message("assistant"):
                                st.markdown(res)
                            st.session_state.chat_history.append({"role": "assistant", "content": res})
                            st.session_state.memory.append({"role": "assistant", "content": res})
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(ai_msg.content)
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_msg.content})
                    
                    save_memory(st.session_state.memory)
                    logger.info(f"✅ 工具调用成功：{prompt[:20]}...")
                except Exception as e:
                    error_msg = f"❌ 工具调用失败：{str(e)}"
                    st.chat_message("assistant").markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    logger.error(error_msg)
        
        # 分支3：普通问答
        else:
            with st.spinner("正在生成回答..."):
                try:
                    client = ZhipuAI(api_key=st.session_state.api_key)
                    response = client.chat.completions.create(
                        model="chatglm_std",
                        messages=st.session_state.memory,
                        temperature=0.1
                    )
                    answer = response.choices[0].message.content
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    # 保存记录
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.memory.append({"role": "assistant", "content": answer})
                    save_memory(st.session_state.memory)
                    logger.info(f"✅ 普通问答成功：{prompt[:20]}...")
                except Exception as e:
                    error_msg = f"❌ 普通问答失败：{str(e)}"
                    st.chat_message("assistant").markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    logger.error(error_msg)

# ====================== 4. 清理临时文件（程序结束时）======================
def cleanup_temp_files():
    """清理上传的PDF临时文件"""
    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.unlink(st.session_state.temp_pdf_path)
            logger.info(f"✅ 临时PDF文件已清理：{st.session_state.temp_pdf_path}")
        except Exception as e:
            logger.error(f"❌ 清理临时文件失败：{str(e)}")

# 页面关闭时清理临时文件
if st.session_state.temp_pdf_path:
    cleanup_temp_files()