import argparse
import json
import time
import os
import yaml
from tqdm import tqdm
from datetime import datetime

# 引入你的组件
from loader import load_data_standardized
from memory_adapter import AgenticMemoryAdapter


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found!")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="Run A-mem Benchmark")
    parser.add_argument("--dataset", type=str, default="hotpot",
                        choices=["hotpot", "mquake", "twiki", "locomo", "longmemeval"],
                        help="Choose which dataset to run")

    # 【新增】Top-K 参数 (默认 None，以便区分是否用户手动指定)
    parser.add_argument("--top_k", type=int, default=None,
                        help="Number of retrieved contexts (overrides config)")

    # 【新增】Limit 参数 (默认 None，以便区分是否用户手动指定)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (overrides config)")

    args = parser.parse_args()

    # 2. 加载配置
    config = load_config("config.yaml")

    dataset_name = args.dataset
    if dataset_name not in config["datasets"]:
        # 兼容性处理：如果 locomo/longmemeval 没在 config 里，可以尝试用默认逻辑或报错
        # 这里假设你已经把它们加进 config.yaml 了
        raise ValueError(f"Dataset '{dataset_name}' not found in config.yaml")

    dataset_config = config["datasets"][dataset_name]
    dataset_path = dataset_config["path"]

    # 【逻辑修改】确定最终的 Limit
    # 优先级：命令行参数 > 配置文件 > 默认值(5)
    if args.limit is not None:
        final_limit = args.limit
    else:
        final_limit = dataset_config.get("limit", 5)

    # 【逻辑修改】确定最终的 Top-K
    # 优先级：命令行参数 > 默认值(5)
    final_top_k = args.top_k if args.top_k is not None else 5

    output_dir = config.get("output_dir", "./results")

    # 3. 初始化适配器
    print(f"Initializing Memory Adapter for [{dataset_name}]...")
    # 注意：现在 adapter 会自动使用 ./temp_memory_workspace 作为工作区
    memory = AgenticMemoryAdapter(config)

    # 4. 加载数据
    print(f"Loading dataset from: {dataset_path}")
    print(f"Settings: Limit={final_limit}, Top-K={final_top_k}")

    samples = load_data_standardized(dataset_path, dataset_type=dataset_name)

    # 应用 Limit
    if final_limit > 0:
        samples = samples[:final_limit]

    # 5. 准备结果容器
    final_output = {
        "dataset": dataset_name,
        "model": config["llm"]["model"],
        "params": {  # 把运行参数也记录到报告里，方便复盘
            "limit": final_limit,
            "top_k": final_top_k
        },
        "timestamp": datetime.now().isoformat(),
        "results": []
    }

    print(f"Running benchmark on {len(samples)} samples...")

    # 获取本次运行的全局时间戳，用于文件夹归类
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 6. 主循环
    for i, sample in tqdm(enumerate(samples), total=len(samples)):

        # 构造一个安全的文件名ID（去除特殊字符）
        safe_id = str(sample['id']).replace("/", "_").replace(" ", "_")

        # === 分支 A: MQuAKE 对比测试 (Pre-edit vs Post-edit) ===
        if sample.get("mode") == "mquake_comparison":

            # --- Phase 1: Pre-edit (原始知识) ---
            memory.clear()
            memory.ingest(sample['pre_edit']['context'])

            start_t1 = time.time()
            # 【关键修改】传入 final_top_k
            res_pre = memory.query(sample['question'], k=final_top_k)
            end_t1 = time.time()

            # 【新增】保存 Pre-edit 快照
            # 命名格式：run时间_步骤_ID_阶段
            snap_name_pre = f"{run_ts}_step_{i + 1:03d}_{safe_id}_PRE"
            memory.save_snapshot(dataset_name, snap_name_pre)

            # --- Phase 2: Post-edit (反事实更新) ---
            memory.clear()
            memory.ingest(sample['post_edit']['context'])

            start_t2 = time.time()
            # 【关键修改】传入 final_top_k
            res_post = memory.query(sample['question'], k=final_top_k)
            end_t2 = time.time()

            # 【新增】保存 Post-edit 快照
            snap_name_post = f"{run_ts}_step_{i + 1:03d}_{safe_id}_POST"
            memory.save_snapshot(dataset_name, snap_name_post)

            # --- Record ---
            result_item = {
                "index": i + 1,
                "id": sample['id'],
                "question": sample['question'],
                "pre_edit": {
                    "gold": sample['pre_edit']['gold'],
                    "pred": res_pre['answer'],
                    "retrieved": res_pre['retrieved'],
                    "latency": round(end_t1 - start_t1, 2)
                },
                "post_edit": {
                    "gold": sample['post_edit']['gold'],
                    "pred": res_post['answer'],
                    "retrieved": res_post['retrieved'],
                    "latency": round(end_t2 - start_t2, 2)
                }
            }

        # === 分支 B: 标准 RAG 测试 (Hotpot/2Wiki) ===
        else:
            memory.clear()
            memory.ingest(sample['contexts_to_ingest'])

            start_time = time.time()
            # 【关键修改】传入 final_top_k
            response = memory.query(sample['question'], k=final_top_k)
            end_time = time.time()

            # 【新增】保存标准测试快照
            snap_name = f"{run_ts}_step_{i + 1:03d}_{safe_id}"
            memory.save_snapshot(dataset_name, snap_name)

            result_item = {
                "index": i + 1,
                "id": sample['id'],
                "question": sample['question'],
                "gold": sample['gold_answer'],
                "pred": response["answer"],
                "retrieved": response["retrieved"],
                "latency": round(end_time - start_time, 2)
            }

        final_output["results"].append(result_item)

    # 7. 保存结果 (修改版：仿 EasyMemory 目录结构)

    # 第一步：拼接子目录，例如 ./results/hotpot/
    dataset_output_dir = os.path.join(output_dir, dataset_name)

    # 第二步：创建这个子目录（如果不存在）
    os.makedirs(dataset_output_dir, exist_ok=True)

    # 第三步：生成文件名 (只需带时间戳即可，因为目录已经区分了数据集)
    out_file = f"{dataset_output_dir}/report_{run_ts}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Done! Report saved to: {out_file}")

if __name__ == "__main__":
    main()