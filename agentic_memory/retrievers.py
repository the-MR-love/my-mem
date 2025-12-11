import json
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from chromadb import EmbeddingFunction
from tenacity import retry, stop_after_attempt, wait_fixed


# --- 新增：自定义 API Embedding 类 ---
class APIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not found. Please install it.")
        # 初始化 OpenAI 客户端用于 Embedding
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    # 【新增】装饰器：如果报错，自动重试 8 次，每次等待 5 秒
    @retry(stop=stop_after_attempt(8), wait=wait_fixed(5))
    def __call__(self, input: List[str]) -> List[List[float]]:
        # 简单预处理：移除换行符
        cleaned_input = [text.replace("\n", " ") for text in input]
        # 调用 API
        resp = self.client.embeddings.create(input=cleaned_input, model=self.model)
        # 确保按顺序返回向量
        return [data.embedding for data in resp.data]


class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""

    def __init__(
            self,
            collection_name: str = "memories",
            embedding_config: Optional[Dict] = None,
            persist_path: Optional[str] = None  # <--- [新增] 接收持久化路径参数
    ):
        """
        初始化 Retriever，强制要求传入 embedding_config 以使用 API
        """

        # === [修改] 开始：根据是否有路径，决定是存内存还是存硬盘 ===
        if persist_path:
            # 如果有路径，使用 PersistentClient，数据会自动落盘
            # 注意：Settings(allow_reset=True) 允许我们在需要时调用 client.reset() 清空数据
            self.client = chromadb.PersistentClient(path=persist_path, settings=Settings(allow_reset=True))
        else:
            # 原有逻辑：纯内存模式
            self.client = chromadb.Client(Settings(allow_reset=True))
        # === [修改] 结束 ===

        if not embedding_config:
            raise ValueError("embedding_config is required for this API-only version.")

        # 使用自定义的 API Embedding Function
        self.embedding_function = APIEmbeddingFunction(
            api_key=embedding_config["api_key"],
            base_url=embedding_config["base_url"],
            model_name=embedding_config["model"]
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        # 将复杂的元数据转换为字符串以便 ChromaDB 存储
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        self.collection.add(
            documents=[document], metadatas=[processed_metadata], ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        # 执行查询
        results = self.collection.query(query_texts=[query], n_results=k)

        # 将元数据转回原始类型（List/Dict）
        if (results is not None) and (results.get("metadatas", [])):
            results["metadatas"] = self._convert_metadata_types(results["metadatas"])

        return results

    def _convert_metadata_types(self, metadatas: List[List[Dict]]) -> List[List[Dict]]:
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, List):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, Dict):
                        self._convert_metadata_dict(metadata_dict)
        return metadatas

    def _convert_metadata_dict(self, metadata: Dict) -> None:
        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    # 尝试将字符串还原为 List 或 Dict
                    # 使用 json.loads 比 ast.literal_eval 更安全，兼容性更好
                    if value.startswith("[") or value.startswith("{"):
                        metadata[key] = json.loads(value)
                except Exception:
                    pass