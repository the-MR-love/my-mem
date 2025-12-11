import json
import os
import shutil
import time
import gc  # <--- [新增] 引入垃圾回收模块
from typing import List, Dict, Any

# 引入 A-mem 系统
from agentic_memory.memory_system import AgenticMemorySystem


class AgenticMemoryAdapter:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化适配器
        :param config: 从 config.yaml 加载的配置字典
        """
        # 1. 提取配置并初始化 A-mem 内核
        # 注意：这里适配了我们之前修改过的支持 API 的 A-mem 构造函数
        embed_cfg = {
            "provider": config["embedding"]["provider"],
            "base_url": config["embedding"]["base_url"],
            "model": config["embedding"]["model"],
            "api_key": config["embedding"]["api_key"]
        }

        # 【新增】从配置中读取进化开关，默认关闭 (False)
        evolution_enabled = config.get("enable_evolution", False)

        # === [策略调整] 沙盒模式 ===
        # 我们使用一个固定的“临时工作区”来运行当前的记忆
        # 这样每次 run 都可以随意 clear 这个工作区，而不会误删存档
        self.workspace_dir = os.path.abspath("./temp_memory_workspace")

        # 确保工作区纯净（启动时清理）
        if os.path.exists(self.workspace_dir):
            try:
                shutil.rmtree(self.workspace_dir)
                time.sleep(0.1)  # 等待文件释放
            except Exception as e:
                print(f"Warning: Could not clear workspace on init: {e}")

        os.makedirs(self.workspace_dir, exist_ok=True)

        self.system = AgenticMemorySystem(
            llm_model=config["llm"]["model"],
            llm_api_key=config["llm"]["api_key"],
            llm_base_url=config["llm"]["base_url"],
            embedding_config=embed_cfg,
            enable_evolution=evolution_enabled,
            persist_dir=self.workspace_dir  # <--- 系统只认这个临时工作区
        )

        # 默认检索 Top-K，也可以在 query 时覆盖
        self.default_k = 3

    def ingest(self, contexts: List[str]):
        """
        [标准接口] 写入记忆
        :param contexts: 文本列表
        """
        for ctx in contexts:
            try:
                self.system.add_note(content=ctx, category="knowledge")
            except Exception as e:
                # [关键修改] 捕获 Collection 丢失错误并尝试自动修复
                if "Collection" in str(e) and "does not exist" in str(e):
                    print(f"⚠️ Warning: ChromaDB collection lost. Attempting repair...")
                    self.system.consolidate_memories()  # 强制重建连接
                    self.system.add_note(content=ctx, category="knowledge")  # 重试
                else:
                    print(f"Error during ingest: {e}")
                    raise e

    def query(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        [标准接口] 提问并获取答案
        :param question: 问题文本
        :return: 包含 'answer' (回答) 和 'retrieved' (检索到的原文列表) 的字典
        """
        if k is None:
            k = self.default_k

        # 1. 检索 (Retrieval)
        # 调用 A-mem 的 search_agentic
        search_results = self.system.search_agentic(question, k=k)

        # 2. 生成 (Generation)
        # 调用内部辅助函数生成答案
        answer_text = self._generate_answer_from_llm(question, search_results)

        # 3. 格式化返回
        # 返回丰富的信息，方便评测脚本记录日志
        return {
            "answer": answer_text,
            "retrieved": [r['content'] for r in search_results]
        }

    def _generate_answer_from_llm(self, question: str, retrieved_contexts: List[Dict]) -> str:
        """
        [内部逻辑] 拼接 Prompt 并调用 LLM
        """
        context_str = "\n".join([f"- {m['content']}" for m in retrieved_contexts])

        prompt = f"""
        Based on the retrieved memories, answer the question concisely.

        Memories:
        {context_str}

        Question: {question}

        REQUIRED OUTPUT FORMAT:
        You must output a single JSON object. Example: {{"answer": "Paris"}}
        """

        # 使用 A-mem 封装好的 LLM Controller
        # 传入简单 schema 触发 JSON 模式
        response_schema = {"type": "json_object"}

        try:
            response_str = self.system.llm_controller.llm.get_completion(
                prompt, response_format=response_schema
            )
            # 清洗可能存在的 markdown 标记
            cleaned_str = response_str.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_str).get("answer", "")
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def clear(self):
        """
        [修改版] 清空临时工作区 (静默处理文件占用问题)
        """
        # 1. 清空内存对象
        self.system.memories.clear()

        try:
            # 2. 尝试断开并重置数据库连接
            # 这一步是逻辑清空，保证数据不可见
            if self.system.retriever:
                try:
                    self.system.retriever.client.reset()
                except:
                    pass
                # 显式解除引用，帮助 GC 识别垃圾
                self.system.retriever.client = None
                self.system.retriever = None

            # === 强制垃圾回收 ===
            gc.collect()
            time.sleep(0.1)  # 稍微等待释放

            # 3. 物理删除文件夹 (静默模式)
            if os.path.exists(self.workspace_dir):
                # ignore_errors=True: 如果遇到 Windows 文件锁 (WinError 32)，直接忽略，不报错，不打印警告。
                # 反正我们已经 reset 过了，残留的物理文件会被覆盖或忽略，不影响逻辑。
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            # 4. 重新创建目录
            os.makedirs(self.workspace_dir, exist_ok=True)

            # 5. 关键：强制 System 重建 Retriever
            # 这一步会创建一个全新的 Client 并连接到目录
            self.system.consolidate_memories()

        except Exception as e:
            print(f"Critical Error in clear(): {e}")

    # === [新增] 对外暴露的快照接口 ===
    def save_snapshot(self, dataset_name: str, snapshot_id: str):
        """
        保存当前记忆的快照到: ./memory_snapshots/{dataset_name}/{snapshot_id}/
        """
        # 定义快照的根目录
        snapshot_root = "./memory_snapshots"

        # 构造完整目标路径
        target_path = os.path.join(snapshot_root, dataset_name, snapshot_id)

        # 调用 System 的底层方法进行复制
        self.system.create_snapshot(target_path)