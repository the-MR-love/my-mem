import json
from pathlib import Path
from typing import List, Dict, Any


def load_data_standardized(file_path: str, dataset_type: str = None) -> List[Dict[str, Any]]:
    """
    统一数据加载入口
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 统一使用 utf-8 读取
    data = json.loads(path.read_text(encoding='utf-8'))

    if not data:
        return []

    # 1. 自动推断类型 (如果未指定)
    if not dataset_type:
        first_item = data[0]
        # 你的 MQuAKE 文件特征：有 case_id 且有 new_single_hops
        if "case_id" in first_item and "new_single_hops" in first_item:
            dataset_type = "mquake"
        else:
            # 2Wiki 和 Hotpot 都有 context 字段，走标准流程
            dataset_type = "hotpot"

    # 2. 分发处理
    if dataset_type.lower() == "mquake":
        return _parse_mquake_comparison(data)  # <--- 改用新的对比解析函数
    else:
        # hotpot, twiki, 2wiki 都走这里
        return _parse_standard_rag(data)


def _parse_mquake_comparison(data: List[Dict]) -> List[Dict[str, Any]]:
    """
    [修正版] MQuAKE 解析逻辑
    核心原则：
    1. Question 不变：始终使用多跳问题 (questions[0])
    2. Context 改变：Pre 用 single_hops，Post 用 new_single_hops
    """
    standardized_data = []
    for item in data:
        # 1. 提取问题 (始终使用同一个多跳问题)
        q_list = item.get("questions", [])
        question = q_list[0] if len(q_list) > 0 else ""

        # 2. Pre-edit 状态 (真实世界)
        pre_context = []
        for hop in item.get("single_hops", []):
            # 拼接成陈述句: "Subject relation Object."
            pre_context.append(f"{hop['cloze']} {hop['answer']}.")

        # 3. Post-edit 状态 (反事实世界)
        post_context = []
        # 使用 new_single_hops，这里面包含了修改后的事实链条
        # 例如: Joe -> Notre Dame (没变) + Notre Dame -> Rugby (变了)
        for hop in item.get("new_single_hops", []):
            post_context.append(f"{hop['cloze']} {hop['answer']}.")

        entry = {
            "mode": "mquake_comparison",
            "id": str(item.get("case_id")),
            "question": question,  # 问题保持一致！
            "pre_edit": {
                "gold": item.get("answer", ""),
                "context": pre_context
            },
            "post_edit": {
                "gold": item.get("new_answer", ""),  # 答案变了！
                "context": post_context
            }
        }
        standardized_data.append(entry)
    return standardized_data


def _parse_standard_rag(data: List[Dict]) -> List[Dict[str, Any]]:
    """
    [回退版] HotpotQA / 2Wiki 处理逻辑
    策略：【段落级拼接】将同一标题下的所有句子拼成一段完整的文本。
    """
    standardized_data = []
    for item in data:
        if "context" in item:
            contexts = []
            for ctx in item["context"]:
                if isinstance(ctx, list) and len(ctx) >= 2:
                    title = ctx[0]
                    # --- 关键修改：拼回去！不要拆！---
                    # 2Wiki 的句子列表用空格连接
                    lines = ctx[1]
                    if isinstance(lines, list):
                        sentences = " ".join(lines)
                    else:
                        sentences = str(lines)
                    contexts.append(f"{title}: {sentences}")
                    # -------------------------------
                elif isinstance(ctx, str):
                    contexts.append(ctx)

            doc_id = item.get("_id") or item.get("id")

            entry = {
                "mode": "standard",
                "id": str(doc_id),
                "question": item.get("question", ""),
                "gold_answer": item.get("answer", ""),
                "contexts_to_ingest": contexts
            }
            standardized_data.append(entry)
    return standardized_data