import logging
import uuid
import json
import pickle  # <--- [新增] 用于对象序列化
import os  # <--- [新增] 用于路径操作
import shutil  # <--- [新增] 用于文件夹复制
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# 引入我们适配好的组件
from .llm_controller import LLMController
from .retrievers import ChromaRetriever

logger = logging.getLogger(__name__)


class MemoryNote:
    """
    记忆节点类：代表系统中的一个最小信息单元。
    包含了内容、元数据以及与其他记忆的链接关系。
    """

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        # 核心内容与 ID
        self.content = content
        self.id = id or str(uuid.uuid4())

        # 语义元数据
        self.keywords = keywords or []
        self.links = links or []  # [关键] 这里存储了图谱结构（Link 到其他记忆的 ID）
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []

        # 时间信息
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        # 统计与进化历史
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []


class AgenticMemorySystem:
    """
    A-mem 核心系统：支持 API 驱动，具备完整的 CRUD 和进化能力。
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',  # 保留参数名以兼容旧配置，实际 API 模式下由 embedding_config 控制
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 llm_api_key: Optional[str] = None,
                 llm_base_url: Optional[str] = None,
                 embedding_config: Optional[Dict] = None,
                 evo_threshold: int = 100,
                 enable_evolution: bool = True,
                 persist_dir: Optional[str] = None):  # <--- [新增] 接收持久化目录参数

        self.memories = {}
        self.model_name = model_name
        self.embedding_config = embedding_config
        self.enable_evolution = enable_evolution
        self.persist_dir = persist_dir  # <--- [新增] 保存目录路径

        # === [新增] 路径准备 ===
        chroma_path = None
        if self.persist_dir:
            # 如果指定了目录，自动创建
            if not os.path.exists(self.persist_dir):
                os.makedirs(self.persist_dir)
            chroma_path = os.path.join(self.persist_dir, "chroma_db")
        # ======================

        # 1. 初始化 ChromaDB
        try:
            # === [修改] 仅在没有持久化需求时才强制重置，否则我们希望加载旧数据 ===
            if not self.persist_dir:
                temp_retriever = ChromaRetriever(collection_name="memories",
                                                 embedding_config=self.embedding_config)
                temp_retriever.client.reset()  # 纯内存模式启动时重置
        except Exception as e:
            logger.warning(f"Could not reset ChromaDB collection: {e}")

        # === [修改] 传入 persist_path 给 Retriever ===
        self.retriever = ChromaRetriever(collection_name="memories",
                                         embedding_config=self.embedding_config,
                                         persist_path=chroma_path)

        # 2. 初始化 LLM
        self.llm_controller = LLMController(
            backend=llm_backend,
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url
        )

        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # === [新增] 尝试从硬盘加载旧的记忆对象 ===
        if self.persist_dir:
            self.load_state()
        # ======================================

        # 3. 完整的进化 Prompt
        self._evolution_system_prompt = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.  

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

Return your decision in JSON format with the following structure:
{{
    "should_evolve": true,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_id_1", "neighbor_memory_id_2"],
    "tags_to_update": ["tag_1", "tag_n"], 
    "new_context_neighborhood": ["new context 1", "new context 2"],
    "new_tags_neighborhood": [["tag_1", "tag_n"], ["tag_1", "tag_n"]]
}}
'''

    # === [新增] 保存状态到磁盘 ===
    def save_state(self):
        """将内存中的 memories 字典保存到 pickle 文件"""
        if not self.persist_dir:
            return

        pkl_path = os.path.join(self.persist_dir, "memories.pkl")
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.memories, f)
            # logger.info(f"Memory state saved to {pkl_path}") # 可选：打印日志
        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")

    # === [新增] 从磁盘加载状态 ===
    def load_state(self):
        """从 pickle 文件加载 memories 字典"""
        if not self.persist_dir:
            return

        pkl_path = os.path.join(self.persist_dir, "memories.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    self.memories = pickle.load(f)
                logger.info(f"Loaded {len(self.memories)} memories from {pkl_path}")
            except Exception as e:
                logger.error(f"Failed to load memory state: {e}")

    # === [新增] 创建快照功能 (核心修改点) ===
    def create_snapshot(self, snapshot_path: str):
        """
        将当前的持久化目录 (persist_dir) 完整复制到 snapshot_path。
        这包括 memories.pkl 和 chroma_db 文件夹。
        """
        if not self.persist_dir:
            logger.warning("Cannot snapshot: No persist_dir configured.")
            return

        # 1. 强制保存一次最新的内存状态，确保 pkl 是最新的
        self.save_state()

        # 2. 确保目标目录存在（如果父目录不存在则创建）
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path, exist_ok=True)

        try:
            # 3. 复制 memories.pkl (对象图谱)
            src_pkl = os.path.join(self.persist_dir, "memories.pkl")
            if os.path.exists(src_pkl):
                shutil.copy2(src_pkl, os.path.join(snapshot_path, "memories.pkl"))

            # 4. 复制 chroma_db 文件夹 (向量索引)
            src_chroma = os.path.join(self.persist_dir, "chroma_db")
            dst_chroma = os.path.join(snapshot_path, "chroma_db")

            # 如果目标已存在（极少情况），先删除，避免 copytree 报错
            if os.path.exists(dst_chroma):
                shutil.rmtree(dst_chroma)

            if os.path.exists(src_chroma):
                # ignore_errors=True 可以忽略 Windows 下某些临时文件被锁定的错误
                shutil.copytree(src_chroma, dst_chroma, ignore=shutil.ignore_patterns('*.lock'))

            logger.info(f"Snapshot created successfully at: {snapshot_path}")

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")

    # ============================

    def analyze_content(self, content: str) -> Dict:
        """
        调用 LLM 分析内容，提取元数据。
        这是记忆生成的必要步骤。
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": ["keyword1", "keyword2"],
                "context": "One sentence summary",
                "tags": ["tag1", "tag2"]
            }

            Content for analysis:
            """ + content

        response_schema = {"type": "json_object"}
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format=response_schema)
            cleaned = response.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """
        [核心入口] 添加笔记 -> (可选进化) -> 存入
        """
        if time is not None:
            kwargs['timestamp'] = time

        # 1. 自动补全元数据
        if not kwargs.get('keywords') or not kwargs.get('context'):
            analysis = self.analyze_content(content)
            kwargs['keywords'] = kwargs.get('keywords') or analysis.get('keywords', [])
            kwargs['context'] = kwargs.get('context') or analysis.get('context', "General")
            kwargs['tags'] = kwargs.get('tags') or analysis.get('tags', [])

        note = MemoryNote(content=content, **kwargs)

        # 2. 进化 (Evolution) - 由开关控制
        evo_label = False
        if self.enable_evolution:
            # 调用核心进化逻辑
            evo_label, note = self.process_memory(note)

        self.memories[note.id] = note

        # 3. 序列化并存入 ChromaDB
        # 注意：ChromaDB 元数据不支持列表，必须 json.dumps 转字符串
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": json.dumps(note.keywords),
            "links": json.dumps(note.links),
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": str(note.evolution_history),
            "category": note.category,
            "tags": json.dumps(note.tags)
        }
        self.retriever.add_document(note.content, metadata, note.id)

        # 触发定期整理 (原版逻辑)
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        # === [新增] 每次添加后自动保存到磁盘 ===
        if self.persist_dir:
            self.save_state()
        # ====================================

        return note.id

    def consolidate_memories(self):
        """重建索引 (维护用)"""
        # === [修改] 重建索引时也需要保持持久化路径 ===
        chroma_path = os.path.join(self.persist_dir, "chroma_db") if self.persist_dir else None

        self.retriever = ChromaRetriever(collection_name="memories",
                                         embedding_config=self.embedding_config,
                                         persist_path=chroma_path)  # <--- [修改] 传入路径

        for memory in self.memories.values():
            metadata = {
                "id": memory.id, "content": memory.content, "keywords": json.dumps(memory.keywords),
                "links": json.dumps(memory.links), "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp, "last_accessed": memory.last_accessed,
                "context": memory.context, "evolution_history": str(memory.evolution_history),
                "category": memory.category, "tags": json.dumps(memory.tags)
            }
            self.retriever.add_document(memory.content, metadata, memory.id)

        # === [新增] 整理完后保存一次 ===
        if self.persist_dir:
            self.save_state()

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """
        [进化辅助] 查找相关记忆并返回格式化字符串。
        此函数被 process_memory 调用，用于给 LLM 提供上下文。
        """
        if not self.memories:
            return "", []
        try:
            results = self.retriever.search(query, k)
            memory_str = ""
            found_ids = []
            if 'ids' in results and results['ids'] and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    if doc_id in self.memories:
                        mem = self.memories[doc_id]
                        # 格式化为文本供 LLM 阅读
                        memory_str += f"memory id:{doc_id}\tcontent: {mem.content}\tcontext: {mem.context}\tkeywords: {mem.keywords}\ttags: {mem.tags}\n"
                        found_ids.append(doc_id)
            return memory_str, found_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """[完整性保留] 返回 raw 格式字符串"""
        return self.find_related_memories(query, k)[0]

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """[完整性保留] 读取单条记忆"""
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs) -> bool:
        """[完整性保留] 更新记忆内容"""
        if memory_id not in self.memories:
            return False
        note = self.memories[memory_id]
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)

        metadata = {
            "id": note.id, "content": note.content, "keywords": json.dumps(note.keywords),
            "links": json.dumps(note.links), "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp, "last_accessed": note.last_accessed,
            "context": note.context, "evolution_history": str(note.evolution_history),
            "category": note.category, "tags": json.dumps(note.tags)
        }
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)

        # === [新增] 更新后自动保存 ===
        if self.persist_dir:
            self.save_state()

        return True

    def delete(self, memory_id: str) -> bool:
        """[完整性保留] 删除记忆"""
        if memory_id in self.memories:
            self.retriever.delete_document(memory_id)
            del self.memories[memory_id]

            # === [新增] 删除后自动保存 ===
            if self.persist_dir:
                self.save_state()

            return True
        return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """[完整性保留] 基础向量搜索"""
        results = self.retriever.search(query, k)
        memories = []
        if not results or 'ids' not in results: return []
        for i, doc_id in enumerate(results['ids'][0]):
            mem = self.memories.get(doc_id)
            if mem:
                memories.append({
                    'id': doc_id,
                    'content': mem.content,
                    'context': mem.context,
                    'keywords': mem.keywords,
                    'score': results['distances'][0][i] if 'distances' in results else 0
                })
        return memories

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        [修复版] 向量检索 + 图谱扩展
        修复了“邻居被截断”的 Bug。现在返回结果数量可能会超过 k。
        """
        if not self.memories:
            return []

        try:
            # 1. 向量检索 (Base Retrieval)
            results = self.retriever.search(query, k)
            memories = []
            seen_ids = set()

            if not results or 'ids' not in results or not results['ids']:
                return []

            # 2. 处理向量结果
            for i, doc_id in enumerate(results['ids'][0]):
                if doc_id in seen_ids: continue

                mem_obj = self.memories.get(doc_id)
                if mem_obj:
                    content = mem_obj.content
                    context = mem_obj.context
                    keywords = mem_obj.keywords
                    tags = mem_obj.tags
                    links = mem_obj.links
                else:
                    meta = results['metadatas'][0][i]
                    content = meta.get('content', '')
                    context = meta.get('context', '')
                    keywords = meta.get('keywords', [])
                    tags = meta.get('tags', [])
                    links = meta.get('links', [])

                memories.append({
                    'id': doc_id,
                    'content': content,
                    'context': context,
                    'keywords': keywords,
                    'tags': tags,
                    'links': links,
                    'is_neighbor': False
                })
                seen_ids.add(doc_id)

            # 3. 图谱扩展 (Graph Expansion)
            # 我们允许在 k 的基础上，额外扩展出一些邻居
            # 比如允许每个向量结果带出它的所有一级连接

            # 创建一个副本进行遍历，防止在循环中修改列表导致的问题
            base_memories = list(memories)

            for memory in base_memories:
                links = memory.get('links', [])
                if isinstance(links, str):
                    try:
                        links = json.loads(links)
                    except:
                        links = []

                for link_id in links:
                    # 只要没见过，就加进来！不要受 k 的限制！
                    if link_id not in seen_ids:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)

            # 🔴 最终返回所有结果 (向量 + 邻居)
            # 不要 [:k]，否则辛苦找来的邻居全没了
            return memories

        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """
        [进化核心] 处理记忆进化
        调用 LLM 判断新记忆是否应该与旧记忆建立连接 (Link) 或更新标签。
        """
        if not self.memories:
            return False, note
        try:
            # 1. 寻找潜在的关联对象 (Vector Top-10)
            neighbors_text, neighbor_ids = self.find_related_memories(note.content, k=10)
            if not neighbors_text:
                return False, note

            # 2. 构造 Prompt
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=str(note.keywords),
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(neighbor_ids)
            )

            response_schema = {"type": "json_object"}
            try:
                # 3. 调用 LLM 决策
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format=response_schema
                )
                cleaned = response.replace("```json", "").replace("```", "").strip()
                response_json = json.loads(cleaned)
                should_evolve = response_json.get("should_evolve", False)

                # 4. 执行进化动作
                if should_evolve:
                    actions = response_json.get("actions", [])
                    for action in actions:
                        if action == "strengthen":
                            # 建立连接
                            suggested = response_json.get("suggested_connections", [])
                            valid_links = [nid for nid in suggested if nid in self.memories]

                            # 1. 正向连接：新记忆 -> 旧记忆
                            if not isinstance(note.links, list): note.links = []
                            note.links.extend(valid_links)

                            # 2. 【新增】反向连接：旧记忆 -> 新记忆
                            # 必须把当前 note.id 加到那些被引用的旧记忆的 links 里
                            for nid in valid_links:
                                neighbor = self.memories[nid]
                                if not isinstance(neighbor.links, list): neighbor.links = []
                                # 避免重复添加
                                if note.id not in neighbor.links:
                                    neighbor.links.append(note.id)
                                    # 这一步很重要：因为我们修改了旧记忆，需要更新 ChromaDB 里的元数据
                                    # 但为了性能，这里可以只在内存改，最后统一 consolidate
                                    # 或者在这里显式调用 update (会慢一点)
                                    # self.update(nid, links=neighbor.links)

                            # 更新标签
                            new_tags = response_json.get("tags_to_update", [])
                            if new_tags: note.tags = new_tags

                        elif action == "update_neighbor":
                            # 反向更新邻居记忆的上下文
                            new_ctxs = response_json.get("new_context_neighborhood", [])
                            new_tags_list = response_json.get("new_tags_neighborhood", [])

                            for idx, nid in enumerate(neighbor_ids):
                                if nid in self.memories:
                                    neighbor_mem = self.memories[nid]
                                    if idx < len(new_ctxs):
                                        neighbor_mem.context = new_ctxs[idx]
                                    if idx < len(new_tags_list):
                                        neighbor_mem.tags = new_tags_list[idx]

                return should_evolve, note
            except Exception as e:
                logger.error(f"Error in evolution execution: {e}")
                return False, note
        except Exception as e:
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note