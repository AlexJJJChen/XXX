import logging
from typing import Dict, Any

from runner.logger import Logger
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
import json

@node_decorator(check_schema_status=False)
def evaluation(task: Any, tentative_schema: Dict[str, Any], execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the predicted SQL queries against the ground truth SQL query.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results.
    """
    logging.info("Starting evaluation")

    ground_truth_sql = task.SQL
    to_evaluate = {
        "candidate_generation": get_last_node_result(execution_history, "candidate_generation"), 
        "revision": get_last_node_result(execution_history, "revision")
    }
    result = {}

    for evaluation_for, node_result in to_evaluate.items():
        predicted_sql = "--"
        evaluation_result = {}

        try:
            if node_result["status"] == "success":
                predicted_sql = node_result["SQL"]
                response = DatabaseManager().compare_sqls(
                    predicted_sql=predicted_sql,
                    ground_truth_sql=ground_truth_sql,
                )

                evaluation_result.update({
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                })
            else:
                evaluation_result.update({
                    "exec_res": "generation error",
                    "exec_err": node_result["error"],
                })
        except Exception as e:
            Logger().log(
                f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
                "error",
            )
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
            })

        evaluation_result.update({
            "Question": task.question,
            "Evidence": task.evidence,
            "GOLD_SQL": ground_truth_sql,
            "PREDICTED_SQL": predicted_sql
        })
        result[evaluation_for] = evaluation_result
    # result_sql_1=DatabaseManager().execute_sql(sql= ground_truth_sql)
    # result_sql_2=DatabaseManager().execute_sql(sql= predicted_sql)
    #print('result_sql_1',result_sql_1)
    #print('result_sql_2',result_sql_2)
    logging.info("Evaluation completed successfully")
    return result



def evaluation_json(task: Any, data, tentative_schema: Dict[str, Any], execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the predicted SQL queries against the ground truth SQL query.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results.
    """
    logging.info("Starting evaluation")

    # ground_truth_sql = task['SQL']
    # bird dataset
    ground_truth_sql = task.SQL

    # spider dataset
    # ground_truth_sql = task.query
    response_sql=eval(data)['SQL']
    to_evaluate = {
        "candidate_generation": response_sql, 
        "revision": response_sql
    }
    result = {}

    # for evaluation_for, node_result in to_evaluate.items():
    predicted_sql = "--"
    evaluation_result = {}

    try:
        # if node_result["status"] == "success":
        predicted_sql = response_sql
        response = DatabaseManager().compare_sqls(
            predicted_sql=predicted_sql,
            ground_truth_sql=ground_truth_sql,
        )

        evaluation_result.update({
            "exec_res": response["exec_res"],
            "exec_err": response["exec_err"],
        })

    except Exception as e:
        Logger().log(
            f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
            "error",
        )
        evaluation_result.update({
            "exec_res": "error",
            "exec_err": str(e),
        })

    evaluation_result.update({
        "Question": task.question,
        "Evidence": task.evidence,
        "GOLD_SQL": ground_truth_sql,
        "PREDICTED_SQL": predicted_sql
    })
    result['candidate_generation'] = evaluation_result
    # result_sql_1=DatabaseManager().execute_sql(sql= ground_truth_sql)
    # result_sql_2=DatabaseManager().execute_sql(sql= predicted_sql)
    #print('result_sql_1',result_sql_1)
    #print('result_sql_2',result_sql_2)
    logging.info("Evaluation completed successfully")

    return result

# file_path="/project/chenjian/nl2sql/CHESS-main/data/dev/dev.json"
# with open(file_path, 'r') as file:
# # Parse JSON data from the file
#     data_gt = json.load(file)

# fp="/project/chenjian/nl2sql/CHESS-main/DeepSeek-MoE/gpt-4o_deepseek_ft.json"
# with open(fp, 'r') as f:
# # Parse JSON data from the file
#     data_ds = json.load(f)

# for i in data_ds:
#     for j in data_gt:
#         if i['question']==j['question']:
#             result=evaluation_json(j,i)