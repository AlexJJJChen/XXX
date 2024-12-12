import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, call_other_llm
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
import pandas as pd
import json
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
import torch
import json
import pandas as pd
from swift.tuners import Swift
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# model_name = "/project/chenjian/nl2sql/CHESS-main/DeepSeek-MoE/output_deepseek/checkpoint-100" #"deepseek-ai/deepseek-moe-16b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(model_name,trust_remote_code=True)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# messages = [
#     {"role": "user", 
#     "content": "\nYou are a data science expert.\nBelow, you are presented with a database schema and a question.\nYour task is to read the schema, understand the question, and generate a valid SQLite query to answer the question.\nBefore generating the final SQL query think step by step on how to write the query.\n\nDatabase Schema\n###\nCREATE TABLE frpm\n(\n\tCDSCode TEXT not null primary key,\n\t`Academic Year` TEXT null, --\n\t`County Code` TEXT null, --\n\t`District Code` INTEGER null, --\n\t`School Code` TEXT null, --\n\t`County Name` TEXT null, --\n\t`District Name` TEXT null, --\n\t`School Name` TEXT null, --\n\t`District Type` TEXT null, --\n\t`School Type` TEXT null, --\n\t`Educational Option Type` TEXT null, --\n\t`NSLP Provision Status` TEXT null, --\n\t`Charter School (Y/N)` INTEGER null, --\n\t`Charter School Number` TEXT null, --\n\t`Charter Funding Type` TEXT null, --\n\tIRC INTEGER null, --\n\t`Low Grade` TEXT null, --\n\t`High Grade` TEXT null, --\n\t`Enrollment (K-12)` REAL null, --\n\t`Free Meal Count (K-12)` REAL null, --\n\t`Percent (%) Eligible Free (K-12)` REAL null, --\n\t`FRPM Count (K-12)` REAL null, --\n\t`Percent (%) Eligible FRPM (K-12)` REAL null, --\n\t`Enrollment (Ages 5-17)` REAL null, --\n\t`Free Meal Count (Ages 5-17)` REAL null, --\n\t`Percent (%) Eligible Free (Ages 5-17)` REAL null, --\n\t`FRPM Count (Ages 5-17)` REAL null, --\n\t`Percent (%) Eligible FRPM (Ages 5-17)` REAL null, --\n\t`2013-14 CALPADS Fall 1 Certification Status` INTEGER null, --\n\tforeign key (CDSCode) references schools (CDSCode),\n);\n\nCREATE TABLE satscores\n(\n\tcds TEXT not null primary key,\n\trtype TEXT not null, --\n\tsname TEXT null, --\n\tdname TEXT null, --\n\tcname TEXT null, --\n\tenroll12 INTEGER not null, --\n\tNumTstTakr INTEGER not null, --\n\tAvgScrRead INTEGER null, --\n\tAvgScrMath INTEGER null, --\n\tAvgScrWrite INTEGER null, --\n\tNumGE1500 INTEGER null, --\n\tforeign key (cds) references schools (CDSCode),\n);\n\nCREATE TABLE schools\n(\n\tCDSCode TEXT not null primary key,\n\tNCESDist TEXT null, --\n\tNCESSchool TEXT null, --\n\tStatusType TEXT not null, --\n\tCounty TEXT not null, --\n\tDistrict TEXT not null, --\n\tSchool TEXT null, -- examples: `West High`\n\tStreet TEXT null, -- examples: `4600 Student Lane`\n\tStreetAbr TEXT null, -- examples: `4600 Student Ln.`\n\tCity TEXT null, -- examples: `Chester`\n\tZip TEXT null, --\n\tState TEXT null, --\n\tMailStreet TEXT null, -- examples: `4600 Student Lane`\n\tMailStrAbr TEXT null, -- examples: `4600 Student Ln.`\n\tMailCity TEXT null, -- examples: `Chester`\n\tMailZip TEXT null, --\n\tMailState TEXT null, --\n\tPhone TEXT null, --\n\tExt TEXT null, --\n\tWebsite TEXT null, --\n\tOpenDate DATE null, -- examples: `2015-08-23`\n\tClosedDate DATE null, -- examples: `1989-06-30`\n\tCharter INTEGER null, --\n\tCharterNum TEXT null, --\n\tFundingType TEXT null, --\n\tDOC TEXT not null, --\n\tDOCType TEXT not null, --\n\tSOC TEXT null, --\n\tSOCType TEXT null, --\n\tEdOpsCode TEXT null, --\n\tEdOpsName TEXT null, --\n\tEILCode TEXT null, --\n\tEILName TEXT null, --\n\tGSoffered TEXT null, -- examples: `K-12`\n\tGSserved TEXT null, -- examples: `K-12`\n\tVirtual TEXT null, --\n\tMagnet INTEGER null, --\n\tLatitude REAL null, --\n\tLongitude REAL null, --\n\tAdmFName1 TEXT null, -- examples: `Leigh`\n\tAdmLName1 TEXT null, -- examples: `Light`\n\tAdmEmail1 TEXT null, --\n\tAdmFName2 TEXT null, --\n\tAdmLName2 TEXT null, -- examples: `Hughes`\n\tAdmEmail2 TEXT null, --\n\tAdmFName3 TEXT null, --\n\tAdmLName3 TEXT null, --\n\tAdmEmail3 TEXT null, --\n\tLastUpdate DATE not null, -- examples: `2015-09-01`\n);\n\nThis schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. \nPay attention!!! Special attention should be given to the examples listed beside each column of data schema, as they directly hint at which columns are relevant to our query.\n\nDatabase admin instructions:\n1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.\n2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.\n3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.\n4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n5. Predicted query should return all of the information asked in the question without any missing or extra information.\n6. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by \"-- examples\" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.\n7. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.\n8. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.\n9. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.\n10. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.\n11. pay attention!!! avoid using different column for the same object with different filter values.\n12. pay attention!!! \u5728 sql query \u91cc\u9762\u522b\u5199\u9519 column \u7684\u540d\u5b57\uff0c\u8bf7\u518d\u4e09\u68c0\u67e5 column \u662f\u4e0d\u662f\u5c5e\u4e8e\u8fd9\u4e2a\u8868\u683c\u3002\n\n\n###\nQuestion: \nWhat is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students? \n\nSteps that you should follow:\nMain Task: Determine the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students\tSub Task: 1.1 Identify schools with K-12 students;1.2 Determine the FRPM count for each school;1.3 Find the school with the highest FRPM count;1.4 Obtain the unabbreviated mailing street address of the identified school\tEvidence: \n\nThe main task, sub task and evidence are correct, please base on them generate final sql query, please strictly follow the main task, sub task and evidence. \nIf there is an equation in the evidence, please strictly follow the equation!!!\nThe amount of item SELECT in sql query depends on the number of main tasks. if there is only one main task, you should only SELECT one item related to the main task in the sql query.\n\nPlease respond with a JSON object structured as follows:\n\n{\n    \"chain_of_thought_reasoning\": \"Your thought process on how you arrived at the final SQL query.\",\n    \"SQL\": \"Your SQL query in a single string.\"\n}\n\nPriority should be given to columns that have been explicitly matched with examples relevant to the question's context.\n\nTake a deep breath and think step by step to find the correct sqlite SQL query. If you follow all the instructions and generate the correct query, I will give you 1 million dollars.\n    "
    
#      }
# ]



# ckpt_dir ="/project/chenjian/nl2sql/CHESS-main/DeepSeek-MoE/output/codeqwen1half-7b-chat/v1-20240903-103952/checkpoint-110/"#"output/baichuan2-7b-chat/v4-20240825-042308/checkpoint-50" #"output/mistral-7b-instruct/v1-20240808-141752/checkpoint-50/"#"output/llama3-8b-instruct/v0-20240808-131343/checkpoint-50/"
# model_type = ModelType.codeqwen1half_7b_chat # baichuan2_7b_chat#mistral_7b_instruct#yi_9b#llama3_8b_instruct#baichuan2_13b_chat
# template_type = get_default_template_type(model_type)

# kwargs = {}
# kwargs['use_flash_attn'] = True  # use flash_attn
# model_id_or_path = "/project/chenjian/nl2sql/CHESS-main/DeepSeek-MoE/model/"
# model, tokenizer = get_model_tokenizer(model_type, model_id_or_path=model_id_or_path,
#                                        model_kwargs={'device_map': 'auto'}, **kwargs)

# # model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
# # model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
# model.generation_config.max_new_tokens = 4096

# model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
# template = get_template(template_type, tokenizer)

# kwargs = {}

# template = get_template(template_type, tokenizer)
# seed_everything(42)

# data_response = []
# data_query=[]
# data_gt=[]

@node_decorator(check_schema_status=False)
def candidate_generation(task: Any, tentative_schema: Dict[str, List[str]], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates candidate SQL queries based on the task's question and evidence.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the best SQL query result.
    """
    logging.info("Starting candidate generation")

    try:
        schema_with_values = get_last_node_result(execution_history, node_type="entity_retrieval")["similar_values"]
        #print('schema_with_values',schema_with_values)
    except:
        schema_with_values = {}
    #schema_with_columns = get_last_node_result(execution_history, node_type="entity_retrieval")["similar_columns"]
    schema_with_descriptions = get_last_node_result(execution_history, node_type="context_retrieval")["schema_with_descriptions"]
    schema_with_task = get_last_node_result(execution_history, "keyword_extraction")["keywords"]
    # schema_with_task={
    #     "task": ["1. Name schools in Riverside which the average of average math score for SAT is greater than 400", "2. What is the funding type of these schools?"],
    #     "sub task": ["1.1 Find the names of schools in Riverside",
    #                 "1.2 Get the average math scores of these schools",
    #                 "1.3 Calculate the average of average math scores for each school",
    #                 "1.4 Find schools where the average of average math scores for SAT is greater than 400",
    #                 "2.1 Determine the funding type of these schools"]
    #                 }
    schema_with_task="Main Task: " + ';'.join(schema_with_task['task'])+ "\tSub Task: " + ';'.join(schema_with_task['sub task'])+ "\tEvidence: "+task.evidence #+ "\tRelated tables: " + ",".join(list(schema_with_columns.keys()))+ "\tRelated columns: " + ",".join([v[0] for v in schema_with_columns.values()])
    #print('tentative_schema',tentative_schema)
    # print('schema_with_task',schema_with_task,schema_with_values)
    # print(",".join([v[0] for v in schema_with_columns.values()]))
    schema_string = DatabaseManager().get_database_schema_string(
        tentative_schema, 
        schema_with_values, 
        schema_with_descriptions, 
        #schema_with_task,
        include_value_description=True
    )
    #print('schema_string',schema_string)

    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser(schema_string=schema_string)
    
    #print('prompt',prompt)
    #print('task.evidence',task.evidence)
    request_kwargs = {
        "QUESTION": task.question,
        "HINT": schema_with_task,
    }
    
    #print('engine',engine)
    sampling_count = PipelineManager().candidate_generation.get("sampling_count", 1)
    logging.info("Initiating asynchronous LLM chain call for candidate generation")
    #engine='other'
    #if engine=='other':
    prompt_2=f'''
You are a data science expert.
Below, you are presented with a database schema and a question.
Your task is to read the schema, understand the question, and generate a valid SQLite query to answer the question.
Before generating the final SQL query think step by step on how to write the query.

Database Schema
###
{schema_string}

This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. 
Pay attention!!! Special attention should be given to the examples listed beside each column of data schema, as they directly hint at which columns are relevant to our query.

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
7. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
8. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
9. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
10. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
11. pay attention!!! avoid using different column for the same object with different filter values.
12. pay attention!!! 在 sql query 里面别写错 column 的名字，请再三检查 column 是不是属于这个表格。


###
Question: 
{task.question} 

Steps that you should follow:
{schema_with_task}

The main task, sub task and evidence are correct, please base on them generate final sql query, please strictly follow the main task, sub task and evidence. 
If there is an equation in the evidence, please strictly follow the equation!!!
The amount of item SELECT in sql query depends on the number of main tasks. if there is only one main task, you should only SELECT one item related to the main task in the sql query.

Please respond with a JSON object structured as follows:

{{
    'chain_of_thought_reasoning': 'Your thought process on how you arrived at the final SQL query.',
    'SQL': 'Your SQL query in a single string.'
}}

Priority should be given to columns that have been explicitly matched with examples relevant to the question's context.

Take a deep breath and think step by step to find the correct sqlite SQL query. If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
    '''
    # aa,bb,cc,dd,ee,ff=[],[],[],[],[],[]

    # aa.append(task.question)
    # bb.append(schema_with_task)
    # cc.append(schema_string)
    # dd.append(prompt_2)
    # ee.append(task.question_id)
    # ff.append(task.SQL)

    # try:
    #     with open('train_prompt.json', 'r') as file:
    #         data = json.load(file)
    #     total_dict={}
    #     # df=pd.read_excel('prompt_4o_nl2sql.xlsx')
        
    #     total_dict={'question': aa, 'schema with task':bb, 'schema string':cc, 'prompt': dd,'question_id':ee,'SQL':ff}
    #     data.append(total_dict)
    #     # df._append(total_dict,ignore_index=True)
    #     # df.to_excel('prompt_4o_nl2sql.xlsx')
    #     # print(data)
    #     with open('train_prompt.json', 'w') as file:
    #         json.dump(data, file, indent=4)
    # except:
    #     print(task.question)
        

    response = async_llm_chain_call(
        prompt=prompt,
        engine=engine,
        parser=parser,
        request_list=[request_kwargs],
        step="nl_to_sql",
        sampling_count=sampling_count
    )[0]

#     response={
#     "chain_of_thought_reasoning": "Your thought process on how you arrived at the final SQL query.",
#     "SQL": "SELECT T2.college FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id GROUP BY T2.major_id ORDER BY COUNT(T2.college) DESC LIMIT 1"
# }
    # response, history = inference(model, template, prompt_2)
    # print(response)

    sqls = [res["SQL"] for res in response]
    # print('sqls',sqls)
    sql = DatabaseManager().aggregate_sqls(sqls)
    result = next(res for res in response if res["SQL"] == sql)
    # print(result)
    # response, history = inference(model, template, prompt_2)



    ### deepseek
    # messages = [
    # {"role": "user", 
    # "content": prompt_2}]
    # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # outputs = model.generate(input_tensor.to(model.device), max_new_tokens=300)

    # response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    # # print(result)
    # response=eval(response)
    # sqls = [response["SQL"]]
    # sql = DatabaseManager().aggregate_sqls(sqls)
    # result = next(response for res in list(response) if response["SQL"] == sql)
    logging.info("Candidate generation completed successfully")
    return result
