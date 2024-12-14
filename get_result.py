import os
import json

def read_json_files_without_dash(directory):
    json_files = []

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        # 检查文件名中是否不包含 "-"
        if '-' not in filename:
            try:
                # 构造完整的文件路径
                file_path = os.path.join(directory, filename)

                # 读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    json_files.append(data)
            except json.JSONDecodeError:
                print(f"文件 {filename} 不是合法的 JSON 文件，跳过该文件。")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误：{e}")

    return json_files

# 指定你的目录路径
directory_path = "/project/chenjian/nl2sql/LAIA-SQL/results/dev/keyword_extraction+entity_retrieval+context_retrieval+candidate_generation+revision+evaluation/dev/2024-12-11-18-23-30"
json_data_list = read_json_files_without_dash(directory_path)

score={"candidate_generation":{"correct":0,"incorrect":0,"error":0,"total":0,"accuracy":0},"revision":{"correct":0,"incorrect":0,"error":0,"total":0,"accuracy":0}}
for json_data in json_data_list:
    candidate_generation=json_data[-1]['candidate_generation']['exec_res']
    revision=json_data[-1]['revision']['exec_res']
    if candidate_generation==1:
        score["candidate_generation"]["correct"]+=1
    elif candidate_generation==0:
        score["candidate_generation"]["incorrect"]+=1
    else:
        score["candidate_generation"]["error"]+=1
    score["candidate_generation"]["total"]=score["candidate_generation"]["correct"]+score["candidate_generation"]["incorrect"]+score["candidate_generation"]["error"]
    score["candidate_generation"]["accuracy"]=score["candidate_generation"]["correct"]/score["candidate_generation"]["total"]  

    if revision==1:
            score["revision"]["correct"]+=1
    elif revision==0:
        score["revision"]["incorrect"]+=1
    else:
        score["revision"]["error"]+=1
    score["revision"]["total"]=score["revision"]["correct"]+score["revision"]["incorrect"]+score["revision"]["error"]
    score["revision"]["accuracy"]=score["revision"]["correct"]/score["revision"]["total"]

print(score)