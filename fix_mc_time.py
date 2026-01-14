import ast

time = "0113_1949"
INPUT_PATH = f"./logs/{time}_mocap_log.txt"
OUTPUT_PATH = f"./logs/{time}_mocap_logf.txt"  # 先输出到新文件，确认无误再覆盖

system_delay = 3393  # ms

with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
    open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        # 把字符串解析成 Python dict
        obj = ast.literal_eval(line)
        
        if len(obj) == 1:
            (k, v), = obj.items()
            if k is None: # 处理 {None: None} 这种
                fout.write(line + "\n")
                continue

            new_k = int(k) + system_delay
            new_obj = {new_k: v}
            fout.write(str(new_obj) + "\n")
        else:
            # 理论上不会发生，你这个文件都是单 key
            fout.write(line + "\n")

print("Done!")
