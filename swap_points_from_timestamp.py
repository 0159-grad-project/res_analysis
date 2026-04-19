import ast
import os
import tempfile
from pathlib import Path

INPUT_FILE = "logs/0409_1257_mocap_log.txt"
OUTPUT_FILE = INPUT_FILE
# OUTPUT_FILE = INPUT_FILE.replace(".txt", "_swapped.txt")

# ====== 从这个时间戳开始生效（包含这一行）======
ts = 1775710658744   # ← 改成你要的那一行时间戳
system_delay = 23659
START_TS = ts - system_delay

# 要交换的两个 index（从 0 开始）
I = 2
J = 6


def process_line(line: str):
    line = line.strip()
    if not line:
        return line, False

    # 你的格式是 Python 字面量：{ts: [[x,y,z], ...]}
    try:
        obj = ast.literal_eval(line)
    except Exception:
        print("⚠️ 解析失败，原样输出：", line[:80])
        return line, False

    if not isinstance(obj, dict) or len(obj) != 1:
        return line, False

    ts = next(iter(obj.keys()))
    points = obj[ts]

    if not isinstance(points, list):
        return line, False

    modified = False

    if ts >= START_TS:
        if len(points) > max(I, J):
            # 交换
            points[I], points[J] = points[J], points[I]
            modified = True

    # 重新格式化输出
    new_line = str({ts: points})
    return new_line, modified


def main():
    changed = 0
    total = 0
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = None

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as fin:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=output_path.parent,
            ) as fout:
                temp_file = fout.name

                for line in fin:
                    total += 1
                    new_line, modified = process_line(line)
                    if modified:
                        changed += 1
                    fout.write(new_line + "\n")
    except Exception:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        raise

    os.replace(temp_file, OUTPUT_FILE)

    print("✅ 处理完成")
    print(f"总行数: {total}")
    print(f"被修改的行数: {changed}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
