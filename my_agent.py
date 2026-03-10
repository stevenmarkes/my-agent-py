import json
import logging
import os
import subprocess
import requests
import sys
from zhipuai import ZhipuAI

# ========== 核心导入 ==========
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI

# ====================== 1. 日志配置 =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== 2. 配置读取 =======================
def load_config():
    """加载智谱API Key配置"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        if not config.get("zhipu_api_key"):
            logger.warning("config.json中未配置智谱API Key")
            print("⚠️ 请先在config.json中填写智谱API Key（https://open.bigmodel.cn/）")
        return config
    except FileNotFoundError:
        logger.info("创建空配置文件config.json")
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"zhipu_api_key": ""}, f, ensure_ascii=False, indent=2)
        print("⚠️ 已创建空配置文件config.json，请填写API Key后重启")
        return {"zhipu_api_key": ""}

# ====================== 3. 对话记忆管理 =======================
def save_memory(memory):
    """保存对话记忆到文件"""
    try:
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        logger.info("对话记忆已保存")
        print("✅ 对话记忆已保存到memory.json")
        return True
    except Exception as e:
        logger.error(f"保存记忆失败：{e}")
        print(f"❌ 保存记忆失败：{e}")
        return False

def load_memory():
    """加载历史对话记忆"""
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            memory = json.load(f)
        logger.info(f"加载{len(memory)}条历史对话")
        return memory
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info("初始化空对话记忆")
        return []

# ====================== 4. 核心工具函数 =======================
def write_file(filename: str, content: str):
    """生成指定文件（过滤非法路径字符）"""
    filename = filename.replace("../", "").replace("/", "").replace("\\", "")
    if not filename:
        return "❌ 文件名无效"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"生成文件：{filename}")
        return f"✅ 已保存文件：{filename}"
    except Exception as e:
        logger.error(f"生成文件失败：{e}")
        return f"❌ 保存失败：{e}"

def read_file(filename: str):
    """读取本地文件内容（前500字）"""
    filename = filename.replace("../", "").replace("/", "").replace("\\", "")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"读取文件：{filename}")
        return f"✅ 读取{filename}成功（前500字）：\n{content[:500]}..."
    except Exception as e:
        logger.error(f"读取文件失败：{e}")
        return f"❌ 读取失败：{e}"

def run_python_code(filename: str):
    """运行本地Python文件（超时15秒）"""
    filename = filename.replace("../", "").replace("/", "").replace("\\", "")
    if not filename.endswith(".py"):
        return "❌ 仅支持运行.py文件"
    
    python_cmd = "python3" if sys.platform != "win32" else "python"
    try:
        result = subprocess.run(
            [python_cmd, filename],
            capture_output=True,
            encoding="utf-8",
            timeout=15
        )
        if result.returncode == 0:
            logger.info(f"运行文件成功：{filename}")
            return f"✅ 运行{filename}成功：\n{result.stdout}"
        else:
            logger.error(f"运行文件失败：{result.stderr}")
            return f"❌ 运行失败：\n{result.stderr}"
    except Exception as e:
        logger.error(f"运行文件异常：{e}")
        return f"❌ 运行异常：{e}"

# ====================== 5. RAG问答链初始化 =======================
def init_rag_chain(zhipu_api_key, pdf_path="test.pdf"):
    """初始化基于PDF的RAG问答链"""
    # 1. 初始化大模型
    llm = ChatZhipuAI(api_key=zhipu_api_key, model="glm-4")
    
    # 2. 加载PDF
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF文件{pdf_path}不存在")
        print(f"⚠️ 未找到{pdf_path}，请将PDF放到项目根目录后重启")
        return None
    
    try:
        documents = PyPDFLoader(pdf_path).load()
        logger.info(f"加载PDF：{len(documents)}页")
        
        # 3. 文档切分+向量库构建
        split_docs = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        ).split_documents(documents)
        
        vector_db = FAISS.from_documents(split_docs, FakeEmbeddings(size=1024))
        
        # 4. 构建RAG链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=False
        )
        logger.info("RAG问答链初始化成功")
        return qa_chain
    except Exception as e:
        logger.error(f"RAG初始化失败：{e}")
        print(f"❌ RAG初始化失败：{e}")
        return None

# ====================== 6. 初始化核心组件 =======================
config = load_config()
api_key = config.get("zhipu_api_key", "")
client = ZhipuAI(api_key=api_key)
memory = load_memory()
rag_chain = init_rag_chain(api_key) if api_key else None

# ====================== 7. 工具配置（仅保留核心工具） =======================
tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "生成/保存文件（用户说“保存为xx文件”时调用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名（如test.py）"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容（用户说“查看xx文件”时调用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "运行Python文件（用户说“运行xx.py”时调用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Python文件名"}
                },
                "required": ["filename"]
            }
        }
    }
]

# ====================== 8. 主交互逻辑 =======================
def main():
    print("===== 🤖 RAG智能助手（求职版）======")
    print("核心功能：PDF问答 | 文件生成/读取/运行 | 对话记忆")
    print("示例：基于test.pdf回答文档内容 | 保存HelloWorld为test.py | 运行test.py")
    print("输入「退出」结束\n")

    while True:
        user_input = input("你：").strip()
        if not user_input:
            print("Agent：请输入有效指令～")
            continue

        # 退出逻辑
        if user_input.lower() in ["退出", "exit", "quit"]:
            save_memory(memory)
            print("Agent：再见！对话记忆已保存～")
            logger.info("用户主动退出")
            break

        # RAG PDF问答（优先处理）
        if rag_chain and ("pdf" in user_input.lower() or "文档" in user_input.lower()):
            try:
                answer = rag_chain.invoke({"query": user_input})["result"]
                print(f"Agent（RAG回答）：{answer}")
                memory.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}])
                logger.info(f"RAG问答成功：{user_input[:20]}")
                continue
            except Exception as e:
                print(f"Agent：RAG问答失败 → {e}")
                logger.error(f"RAG问答失败：{e}")

        # 工具调用/普通问答
        memory.append({"role": "user", "content": user_input})
        try:
            # 调用智谱API处理指令
            response = client.chat.completions.create(
                model="chatglm_std",
                messages=memory,
                tools=tools,
                tool_choice="auto",
                temperature=0.1
            )
            ai_msg = response.choices[0].message
            memory.append({"role": "assistant", "content": ai_msg.content})

            # 处理工具调用
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
                    
                    print(f"Agent：{res}")
                    memory.append({"role": "assistant", "content": res})
            else:
                # 普通问答
                print(f"Agent：{ai_msg.content}")

        except requests.exceptions.ConnectionError:
            print("Agent：网络异常，请检查网络连接～")
            memory.pop()
        except Exception as e:
            if "AuthenticationError" in str(e):
                print("Agent：API Key错误/未实名，请检查～")
            else:
                print(f"Agent：出错了 → {e}（详情见agent.log）")
            memory.pop()
            logger.error(f"交互异常：{e}", exc_info=True)

if __name__ == "__main__":
    main()