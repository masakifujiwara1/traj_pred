# ファイルを読み込んでデータを取得
file_path = "./dataset_org/zara02/obsmat.txt"  # ファイルのパスを指定してください

with open(file_path, "r") as file:
    lines = file.readlines()

# 各行を適切に分割して小数点以下2桁に丸める
rounded_lines = []
for line in lines:
    values = line.split()  # 空白文字で分割
    del values[3]
    del values[5]
    rounded_values = [f"{float(val):.2f}" for val in values]  # 小数点以下2桁に丸める
    rounded_line = "	".join(rounded_values)
    rounded_lines.append(rounded_line)

# 新しいファイルに書き出す
rounded_file_path = "./round_dataset/zara02/rounded.txt"  # 書き出すファイルのパスを指定してください

with open(rounded_file_path, "w") as rounded_file:
    for line in rounded_lines:
        rounded_file.write(line + "\n")

print("データを小数点以下2桁に丸めて新しいファイルに書き出しました。")
