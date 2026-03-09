# AI Agent 智能体（求职版）- 整合工程化+多工具+持久化
# 技术栈：Python + 智谱GLM API + 日志系统 + 配置管理 + 异常处理
import json
import logging
import subprocess
import requests
from zhipuai import ZhipuAI

# ====================== 1. 日志配置（工程化核心）======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),  # 日志文件（排查问题用）
        logging.StreamHandler()  # 终端实时输出
    ]
)
logger = logging.getLogger(__name__)

# ====================== 2. 配置读取（安全+规范）======================
def load_config():
    """加载配置文件，避免API Key硬编码"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        # 校验配置完整性
        if not config.get("zhipu_api_key"):
            logger.warning("配置文件中未填写智谱API Key")
            print("⚠️ 请先在config.json中填写你的智谱API Key！")
        return config
    except FileNotFoundError:
        logger.error("配置文件config.json不存在，已自动创建空配置")
        # 自动创建空配置文件
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({"zhipu_api_key": ""}, f, ensure_ascii=False, indent=2)
        return {"zhipu_api_key": ""}

# ====================== 3. 对话记忆持久化（功能增强）======================
def save_memory(memory):
    """保存对话记忆到文件，重启不丢失"""
    try:
        with open("memory.json", "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        logger.info("对话记忆已保存到memory.json")
    except Exception as e:
        logger.error(f"保存记忆失败：{str(e)}")

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

# ====================== 4. 多工具定义======================
def write_file(filename: str, content: str):
    """生成指定文件（核心工具）"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"成功生成文件：{filename}")
        return f"✅ 已自动保存文件：{filename}"
    except PermissionError:
        logger.error(f"无权限写入{filename}")
        return f"❌ 保存失败：没有文件写入权限，请检查文件夹权限"
    except Exception as e:
        logger.error(f"生成{filename}失败：{str(e)}")
        return f"❌ 保存{filename}失败：{str(e)}"

def read_file(filename: str):
    """读取本地文件内容（扩展工具）"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"成功读取文件：{filename}")
        # 只显示前500字，避免输出过长
        return f"✅ 读取{filename}成功（前500字）：\n{content[:500]}..."
    except FileNotFoundError:
        logger.error(f"文件{filename}不存在")
        return f"❌ 读取失败：{filename}文件不存在"
    except Exception as e:
        logger.error(f"读取{filename}失败：{str(e)}")
        return f"❌ 读取{filename}失败：{str(e)}"

def run_python_code(filename: str):
    """运行本地Python文件（扩展工具）"""
    try:
        # 超时保护
        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            encoding="utf-8",
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"成功运行{filename}")
            return f"✅ 运行{filename}成功：\n{result.stdout}"
        else:
            logger.error(f"运行{filename}失败：{result.stderr}")
            return f"❌ 运行{filename}失败：\n{result.stderr}"
    except subprocess.TimeoutExpired:
        logger.error(f"运行{filename}超时（10秒）")
        return f"❌ 运行{filename}超时：代码运行超过10秒，请检查是否有死循环"
    except FileNotFoundError:
        logger.error(f"运行失败：{filename}文件不存在")
        return f"❌ 运行失败：{filename}文件不存在"
    except Exception as e:
        logger.error(f"运行{filename}异常：{str(e)}")
        return f"❌ 运行{filename}异常：{str(e)}"

# ====================== 5. 初始化客户端+记忆 ======================
config = load_config()
client = ZhipuAI(api_key=config["zhipu_api_key"])
memory = load_memory()  # 加载历史记忆

# ====================== 6. 工具配置（智谱API规范）======================
tools = [
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
]

# ====================== 7. 主运行逻辑 ======================
def main():
    print("===== 🤖 AI Agent 智能体（求职版）======")
    print("📌 支持功能：生成代码/读取文件/运行代码/上下文记忆")
    print("📌 示例指令：")
    print("   - 帮我写本地问答机器人，保存为qarobot.py")
    print("   - 查看qarobot.py")
    print("   - 运行qarobot.py")
    print("🔑 输入「退出」结束对话\n")

    while True:
        user_input = input("你：").strip()
        if not user_input:
            print("Agent：请输入有效的指令～")
            continue

        # 退出逻辑（保存记忆）
        if user_input.lower() in ["退出", "exit", "quit"]:
            save_memory(memory)
            print("Agent：再见！对话记忆已保存到memory.json～")
            logger.info("用户主动退出，程序正常结束")
            break

        # 加入对话记忆
        memory.append({"role": "user", "content": user_input})

        try:
            # 调用智谱GLM大模型
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
                    # 解析工具参数（容错处理）
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.error("AI返回参数格式错误，无法解析")
                        print("Agent：⚠️ 工具调用失败：AI返回参数格式异常")
                        continue

                    # 执行对应工具
                    if func_name == "write_file":
                        res = write_file(args.get("filename", ""), args.get("content", ""))
                    elif func_name == "read_file":
                        res = read_file(args.get("filename", ""))
                    elif func_name == "run_python_code":
                        res = run_python_code(args.get("filename", ""))
                    else:
                        res = f"❌ 不支持的工具：{func_name}"
                        logger.warning(f"调用未知工具：{func_name}")
                    
                    print(f"Agent：{res}")
                    # 把工具执行结果加入记忆
                    memory.append({"role": "assistant", "content": res})
            else:
                # 普通对话回复
                print(f"Agent：{ai_msg.content}")

        # 分类异常处理（鲁棒性核心）
        except requests.exceptions.ConnectionError:
            logger.error("网络异常：无法连接智谱API")
            print("Agent：⚠️ 网络出问题了，请检查网络连接后重试")
            memory.pop()  # 移除本次错误的用户输入，避免记忆污染
        except zhipuai.core._errors.APIAuthenticationError:
            logger.error("API认证失败：密钥错误/未实名认证")
            print("Agent：⚠️ 身份验证失败！请检查：")
            print("   1. config.json中的API Key是否正确")
            print("   2. 智谱账号是否完成实名认证")
            memory.pop()
        except Exception as e:
            logger.error(f"未知错误：{str(e)}", exc_info=True)
            print("Agent：⚠️ 出错了！详情请查看agent.log日志文件")
            memory.pop()

if __name__ == "__main__":
    main()