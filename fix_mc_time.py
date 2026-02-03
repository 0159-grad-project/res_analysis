import ast

time = "0122_1855"
PATH = f"./logs/{time}_mocap_log.txt"

system_delay = 3765  # ms

# 1) 先读入所有行
with open(PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []

# 2) 处理
for line in lines:
    line = line.strip()
    if not line:
        new_lines.append("\n")
        continue

    # 把字符串解析成 Python dict
    obj = ast.literal_eval(line)

    if len(obj) == 1:
        (k, v), = obj.items()
        if k is None:  # 处理 {None: None} 这种
            new_lines.append(line + "\n")
            continue

        new_k = int(k) + system_delay
        new_obj = {new_k: v}
        new_lines.append(str(new_obj) + "\n")
    else:
        # 理论上不会发生
        new_lines.append(line + "\n")

# 3) 覆盖写回同一个文件
with open(PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Done!")
