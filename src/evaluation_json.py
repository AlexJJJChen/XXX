import logging
from typing import Dict, Any

from runner.logger import Logger
# from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
import json
from tqdm import tqdm

# ft_v2 0.30
# ft_v3 0.55
# revision granite 0.618


from database_utils.execution import execute_sql, compare_sqls, validate_sql_query, aggregate_sqls

file_path="/project/chenjian/nl2sql/CHESS-main/data/dev/dev.json"
with open(file_path, 'r') as file:
# Parse JSON data from the file
    data_gt = json.load(file)

fp="/project/chenjian/nl2sql/CHESS-main/DeepSeek-MoE/gpt-4o_deepseek_v3_revision.json"
with open(fp, 'r') as f:
# Parse JSON data from the file
    data_ds = json.load(f)
a=0
b=0
for data in tqdm(data_ds):
    for task in data_gt:
        # print(data)
        if data['question']==task['question']:
            # print(1)
            # print(data['response'].split('"revised_SQL\":')[-1].split("}")[0])
            try:
                logging.info("Starting evaluation")

                ground_truth_sql = task['SQL']

                # response_sql=data['response'].split('"revised_SQL\":')[-1].split("}")[0]  #for ft v3 model
                # print(result)
                response_sql=eval(data['response'])['revised_SQL'] #['SQL'] #['revised_SQL']
                to_evaluate = {
                    "candidate_generation": response_sql, 
                    "revision": response_sql
                }
                result = {}
                # print("ground_truth_sql",ground_truth_sql)
                # print("response_sql",response_sql)

                # for evaluation_for, node_result in to_evaluate.items():
                predicted_sql = "--"
                evaluation_result = {}

            
                # if node_result["status"] == "success":
                db_path ="/project/chenjian/nl2sql/CHESS-main/data/dev/dev_databases/"+f"{task['db_id']}/"+ f"{task['db_id']}.sqlite"
                # print(db_path)
                predicted_sql = response_sql.split(";")[0]
                response = compare_sqls(
                    db_path=db_path,
                    predicted_sql=predicted_sql,
                    ground_truth_sql=ground_truth_sql,
                )

                evaluation_result.update({
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                })

            except Exception as e:
                # Logger().log(
                #     f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
                #     "error",
                # )
                # evaluation_result.update({
                #     "exec_res": "error",
                #     "exec_err": str(e),
                # })
                # print(1)
                # b+=1

                continue

            evaluation_result.update({
                "Question": task['question'],
                "Evidence": task['evidence'],
                "GOLD_SQL": ground_truth_sql,
                "PREDICTED_SQL": predicted_sql
            })
            result['candidate_generation'] = evaluation_result
            # result_sql_1=execute_sql(db_path=db_path, sql= ground_truth_sql)
            # result_sql_2=execute_sql(db_path=db_path, sql= predicted_sql)
            #print('result_sql_1',result_sql_1)
            #print('result_sql_2',result_sql_2)
            logging.info("Evaluation completed successfully")
            if evaluation_result['exec_res']==1:
                a+=1
            
            else:
                if evaluation_result['exec_err']!='incorrect answer':
                    b+=1
            # print(result)
            # print(result_sql_1, result_sql_2)
print("correct:",a,"incorrect:", len(data_ds)-a-b, "error:", b)
print("accuracy:", a/(len(data_ds)-b))



# import pandas as pd


# file_path="/project/chenjian/nl2sql/CHESS-main/data/dev/dev.json"
# with open(file_path, 'r') as file:
# # Parse JSON data from the file
#     data_gt = json.load(file)

# fp="/project/chenjian/nl2sql/CHESS-main/granite_instruct_drop_error.xlsx"
# # with open(fp, 'r') as f:
# # # Parse JSON data from the file
# data_ds = pd.read_excel(fp)
# a=0
# b=0
# # print(data_ds[:10])
# for i in tqdm(range(len(data_ds))):
#     # data=data_ds[i]
#     for task in data_gt:
#         # print(data_ds['cg_p'][i].split('Question:')[-1].split('\nSteps that you should follow:')[0])
#         if data_ds['cg_p'][i].split('Question:')[-1].split('\nSteps that you should follow:')[0].strip()==task['question']:
#             try:
#                 logging.info("Starting evaluation")

#                 ground_truth_sql = task['SQL']
#                 # start_index = data['response'].find('"SQL":')
#                 # end_index = data['response'].rfind('}')
#                 # response_sql = data['response'][start_index+6:end_index]
#                 # print(result)
#                 response_sql=data_ds['cg_sql'][i]
#                 to_evaluate = {
#                     "candidate_generation": response_sql, 
#                     "revision": response_sql
#                 }
#                 result = {}

#                 # for evaluation_for, node_result in to_evaluate.items():
#                 predicted_sql = "--"
#                 evaluation_result = {}

            
#                 # if node_result["status"] == "success":
#                 db_path ="/project/chenjian/nl2sql/CHESS-main/data/dev/dev_databases/"+f"{task['db_id']}/"+ f"{task['db_id']}.sqlite"
#                 # print(db_path)
#                 predicted_sql = response_sql.split(";")[0]
#                 response = compare_sqls(
#                     db_path=db_path,
#                     predicted_sql=predicted_sql,
#                     ground_truth_sql=ground_truth_sql,
#                 )

#                 evaluation_result.update({
#                     "exec_res": response["exec_res"],
#                     "exec_err": response["exec_err"],
#                 })

#             except Exception as e:
#                 # Logger().log(
#                 #     f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
#                 #     "error",
#                 # )
#                 # evaluation_result.update({
#                 #     "exec_res": "error",
#                 #     "exec_err": str(e),
#                 # })
#                 # print(1)
#                 # b+=1

#                 continue

#             evaluation_result.update({
#                 "Question": task['question'],
#                 "Evidence": task['evidence'],
#                 "GOLD_SQL": ground_truth_sql,
#                 "PREDICTED_SQL": predicted_sql
#             })
#             result['candidate_generation'] = evaluation_result
#             # result_sql_1=execute_sql(db_path=db_path, sql= ground_truth_sql)
#             # result_sql_2=execute_sql(db_path=db_path, sql= predicted_sql)
#             #print('result_sql_1',result_sql_1)
#             #print('result_sql_2',result_sql_2)
#             logging.info("Evaluation completed successfully")
#             if evaluation_result['exec_res']==1:
#                 a+=1
            
#             else:
#                 if evaluation_result['exec_err']!='incorrect answer':
#                     b+=1
#             # print(result)
#             # print(result_sql_1, result_sql_2)
# print("correct:",a,"incorrect:", len(data_ds)-a-b, "error:", b)
# print("accuracy:", a/(len(data_ds)-b))