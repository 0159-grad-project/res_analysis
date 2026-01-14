import ast

INPUT_FILE = "logs/0108_1843_mocap_log.txt"
OUTPUT_FILE = "0108_1843_mocap_log.txt"

TARGET = (-203, -2141, 388)
TOL = 2

def is_bad_point(p):
    if p is None:
        return False
    x, y, z = p
    return (
        abs(x - TARGET[0]) <= TOL and
        abs(y - TARGET[1]) <= TOL and
        abs(z - TARGET[2]) <= TOL
    )

def line_should_be_removed(line: str) -> bool:
    try:
        data = ast.literal_eval(line.strip())
    except Exception:
        # 解析失败的行你可以选择：
        # return True  # 直接删
        # 或
        return False   # 保留
    # data: {timestamp: [ [x,y,z], ... ]}
    for _, points in data.items():
        if points is None:
            return False
        for p in points:
            if is_bad_point(p):
                return True
    return False

removed = 0
kept = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
    open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        if line_should_be_removed(line):
            removed += 1
        else:
            fout.write(line)
            kept += 1

print(f"完成：删除 {removed} 行，保留 {kept} 行")
