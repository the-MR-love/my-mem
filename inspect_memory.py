import pickle
import os
import sys

# 设置您的存储路径
MEMORY_PATH = "./temp_memory_workspace/memories.pkl"


def inspect():
    # 1. 检查文件是否存在
    if not os.path.exists(MEMORY_PATH):
        print(f"❌ 错误：找不到文件 {MEMORY_PATH}")
        print("请先运行 run.py 生成记忆后再来查看。")
        return

    print(f"🔍 正在读取记忆文件: {MEMORY_PATH} ...")

    try:
        # 2. 使用二进制读取模式 'rb' 加载
        with open(MEMORY_PATH, 'rb') as f:
            memories = pickle.load(f)

        # 3. 打印统计信息
        print(f"✅ 读取成功！")
        print(f"📊 当前大脑中共有 【{len(memories)}】 条记忆片段。")
        print("=" * 50)

        # 4. 展示记忆详情（如果有的话）
        for i, (mem_id, note) in enumerate(memories.items()):

            print(f"🧠 记忆 ID: {mem_id}")
            print(f"📝 内容摘要: {note.content[:100]}..." if len(note.content) > 100 else f"📝 内容: {note.content}")
            print(f"🏷️ 关键词: {note.keywords}")
            print(f"🔗 链接关系 (Links): {note.links}")
            print(f"🏷️ 标签: {note.tags}")
            print(f"⏱️ 记录时间: {note.timestamp}")
            print("-" * 50)


    except Exception as e:
        print(f"❌ 读取失败: {e}")
        print("可能原因：文件损坏，或者这不是一个有效的 pickle 文件。")


if __name__ == "__main__":
    inspect()