import ast

INPUT_FILE = "logs\\0113_1949_mocap_log.txt"
OUTPUT_FILE = "data_swapped.log"

# ====== 从这个时间戳开始生效（包含这一行）======
START_TS = 1768304988047   # ← 改成你要的那一行时间戳

# 要交换的两个 index（从 0 开始）
I = 2
J = 5


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

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            new_line, modified = process_line(line)
            if modified:
                changed += 1
            fout.write(new_line + "\n")

    print("✅ 处理完成")
    print(f"总行数: {total}")
    print(f"被修改的行数: {changed}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
