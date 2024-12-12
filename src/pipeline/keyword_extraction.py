import logging
from typing import Any, Dict

from llm.models import async_llm_chain_call
from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager
from openai import OpenAI
import pandas as pd
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
import torch
import json
import pandas as pd
from swift.tuners import Swift
from tqdm import tqdm

# ckpt_dir = "/project/chenjian/nl2sql/CHESS-main/nl2sql/output/mistral-7b-instruct/v3-20240825-224321/checkpoint-83/" #"nl2sql/output/baichuan2-7b-chat/v2-20240823-122053/checkpoint-62/"#"output/llama3-8b-instruct/v0-20240808-131343/checkpoint-50/"
# model_type = ModelType.mistral_7b_instruct #yi_9b#llama3_8b_instruct#baichuan2_13b_chat
# template_type = get_default_template_type(model_type)

# kwargs = {}
# kwargs['use_flash_attn'] = True  # use flash_attn
# model_id_or_path = "/project/chenjian/nl2sql/CHESS-main/AI-ModelScope/Mistral-7B-Instruct-v0___1/"
# model, tokenizer = get_model_tokenizer(model_type, model_id_or_path=model_id_or_path,
#                                        model_kwargs={'device_map': 'auto'}, **kwargs)

# # model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
# model.generation_config.max_new_tokens = 4096

# model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
# template = get_template(template_type, tokenizer)

# kwargs = {}

# template = get_template(template_type, tokenizer)
# seed_everything(42)
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  
    api_version="2024-08-01-preview"
)


def gpt_keyword(task):
#     completion = client.chat.completions.create(
#     model = "gpt-4o-2024-08-06",#"gpt-4-turbo-2024-04-09",#"gpt-3.5-turbo-0125",#"gpt-4o-all", #"gpt-4o-2024-05-13",# "claude-3-opus-20240229"
#     messages=[
#     {"role": "system", "content": f'''you are a professional english teacher.
# question: {task.question}'''},
#     {"role": "user", "content": '''

# 1. 上面的句子肯定是没有错误的。拆解上面的句子，拆分成主线任务和支线任务。
# 2. 告诉我每个 sub tasks 要做什么操作，拆解成 object 和 implementation，只能够给我输出句子里的关于 object 和 implement 的 keywords，不要多余的词。

# ### example one:
# Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?
# Please only respond with a JSON object structured as follows, don't change the keys name:
# {
# 'question':"Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?",
# 'task':["1. Name schools in Riverside which the average of average math score for SAT is grater than 400", "2. what is the funding type of these schools?"],
# 'sub task':["1.1 find the name of schools in Riverside",
#             "1.2 get the average math score of these school", 
#             "1.3 calculate the average score of average math score of eah school.", 
#             "1.4 find the school which the average of average math score for SAT is grater than 400",
#             "2.1 the funding type of these schools"],
# 'object':['Name schools','funding type', 'average math score for SAT','schools'],
# 'implementation':[{'in':'Riverside'}, {'is grater than':'400'}]
# }
# ### example two:
# {
# 'question': "How many units of item no.9 were sold in store no.1 in total in January, 2012?",
# 'task': ["Determine the total units sold of item no.9 in store no.1 in January, 2012"],
# 'sub task': ["1.1 Identify store no.1",
#                 "1.2 Identify item no.9",
#                 "1.3 Track sales in January, 2012",
#                 "1.4 Calculate total units sold of item no.9"],
# 'object': ['units', 'item no', 'store no'],
# 'implementation': [{'store no.': '1'},
#                     {'item no.': '9'},
#                     {'in': 'January, 2012'}]
# }

# ### example three:
# {
# "question": "List the names of schools with more than 30 difference in enrollments between K-12 and ages 5-17? Please also give the full street address of the schools.",
# "task": ["1. List the names of schools with more than 30 difference in enrollments between K-12 and ages 5-17",
#          "2. Provide the full street address of the schools"],
# "sub task": ["1.1 Identify enrollments for K-12 in schools",
#                 "1.2 Identify enrollments for ages 5-17 in schools",
#                 "1.3 Calculate the difference in enrollments between K-12 and ages 5-17",
#                 "1.4 List schools where the difference in enrollments is more than 30",
#                 "2.1 Get the full street address of these schools"],
# "object": ["names of schools", "difference in enrollments", "full street address"],
# "implementation": [{"between": "K-12 and ages 5-17"},
#                 {"is more than": "30"}]
# }
     
# object 就是 question 里面的关键词。
# implementation 的字典里面的 value 一般都是一两个词，如果你选的 value 包括很多词的话，再次思考它属不属于 filter 的条件，然后修改。要么是数字，要么是形容词。

#     '''
#         }

#         ]
#         )
    messages = [
    (
        "system",
        f'''you are a professional english teacher.
question: {task.question}''',
    ),
    ("human", '''
1. 上面的句子肯定是没有错误的。拆解上面的句子，拆分成主线任务和支线任务。
2. 告诉我每个 sub tasks 要做什么操作，拆解成 object 和 implementation，只能够给我输出句子里的关于 object 和 implement 的 keywords，不要多余的词。

### example one:
Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?
Please only respond with a Dict structured as follows, don't change the keys name:
{
'question':"Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?",
'task':["1. Name schools in Riverside which the average of average math score for SAT is grater than 400", "2. what is the funding type of these schools?"],
'sub task':["1.1 find the name of schools in Riverside",
            "1.2 get the average math score of these school", 
            "1.3 calculate the average score of average math score of eah school.", 
            "1.4 find the school which the average of average math score for SAT is grater than 400",
            "2.1 the funding type of these schools"],
'object':['Name schools','funding type', 'average math score for SAT','schools'],
'implementation':[{'in':'Riverside'}, {'is grater than':'400'}]
}
### example two:
{
'question': "How many units of item no.9 were sold in store no.1 in total in January, 2012?",
'task': ["Determine the total units sold of item no.9 in store no.1 in January, 2012"],
'sub task': ["1.1 Identify store no.1",
                "1.2 Identify item no.9",
                "1.3 Track sales in January, 2012",
                "1.4 Calculate total units sold of item no.9"],
'object': ['units', 'item no', 'store no'],
'implementation': [{'store no.': '1'},
                    {'item no.': '9'},
                    {'in': 'January, 2012'}]
}

### example three:
{
"question": "List the names of schools with more than 30 difference in enrollments between K-12 and ages 5-17? Please also give the full street address of the schools.",
"task": ["1. List the names of schools with more than 30 difference in enrollments between K-12 and ages 5-17",
         "2. Provide the full street address of the schools"],
"sub task": ["1.1 Identify enrollments for K-12 in schools",
                "1.2 Identify enrollments for ages 5-17 in schools",
                "1.3 Calculate the difference in enrollments between K-12 and ages 5-17",
                "1.4 List schools where the difference in enrollments is more than 30",
                "2.1 Get the full street address of these schools"],
"object": ["names of schools", "difference in enrollments", "full street address"],
"implementation": [{"between": "K-12 and ages 5-17"},
                {"is more than": "30"}]
}
     
object 就是 question 里面的关键词。
implementation 的字典里面的 value 一般都是一两个词，如果你选的 value 包括很多词的话，再次思考它属不属于 filter 的条件，然后修改。要么是数字，要么是形容词。
最终只需要输出 dict。
    '''),
    ]
    keywords = llm.invoke(messages).content
    print(keywords)
    # keywords=completion.choices[0].message.content
        # print(keywords)
    # 寻找第一个'{'的索引
    start_index = keywords.find('{')

    # 寻找最后一个'}'的索引
    end_index = keywords.rfind('}')

    # 提取这两个索引之间的内容，包括大括号本身
    keywords = keywords[start_index:end_index + 1]

    return keywords


@node_decorator(check_schema_status=False)
def keyword_extraction(task: Any, tentative_schema: Dict[str, Any], execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts keywords from the task using an LLM chain call.

    Args:
        task (Any): The task object containing the evidence and question.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted keywords.
    """
    request_kwargs = {
        "QUESTION": task.question,
    }
    
    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser()
    
    # api_key = "sk-hMHksPxJPTpRZbTk9fD2D5173a1c4108BdB1A340A34eE640"
    # api_base = "https://api.xhrai.com/v1"
    # client = OpenAI(api_key=api_key, base_url=api_base)

    #print("#####")
    try:
        df_2=pd.read_excel('xxx/project/chenjian/nl2sql/CHESS-main/data/dev/mistral_7b_keywords.xlsx') #gpt_4o_keywords
        df=pd.read_excel('xxx/project/chenjian/nl2sql/CHESS-main/data/dev/gpt_4o_keywords.xlsx') #gpt_4o_keywords
        question_list=list(df['question'])
        index = question_list.index(str(task.question))
        keywords=df['keyword'][index]
        print(1)
    
        question_list_2=list(df_2['question'])
        index_2 = question_list_2.index(str(task.question))
        keywords_2=df_2['keyword'][index_2]

    #         completion = client.chat.completions.create(
    #         model = "gpt-4o-2024-05-13",#"gpt-4-turbo-2024-04-09",#"gpt-3.5-turbo-0125",#"gpt-4o-all", #"gpt-4o-2024-05-13",# "claude-3-opus-20240229"
    #         messages=[
    #         {"role": "system", "content": f'''you are a professional english teacher.
    #         question: {task.question}'''},
    #         {"role": "user", "content": '''

    # 1. 上面的句子肯定是没有错误的。拆解上面的句子，拆分成主线任务和支线任务。
    # 2. 告诉我每个 sub tasks 要做什么操作，拆解成 object 和 implementation，只能够给我输出句子里的关于 object 和 implement 的 keywords，不要多余的词。

    # ### example one:
    # Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?
    # Please only respond with a JSON object structured as follows, don't change the keys name:
    # {
    # 'question':"Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?",
    # 'task':["1. Name schools in Riverside which the average of average math score for SAT is grater than 400", "2. what is the funding type of these schools?"],
    # 'sub task':["1.1 find the name of schools in Riverside",
    #             "1.2 get the average math score of these school", 
    #             "1.3 calculate the average score of average math score of eah school.", 
    #             "1.4 find the school which the average of average math score for SAT is grater than 400",
    #             "2.1 the funding type of these schools"],
    # 'object':['Name schools'],
    # 'implementation':[{'in':'Riverside'}, {'the average of':'average math score for SAT'},{'is grater than':'400'},{'':'funding type'}]
    # }
    # ### example two:
    # {
    # 'question': "How many units of item no.9 were sold in store no.1 in total in January, 2012?",
    # 'task': ["Determine the total units sold of item no.9 in store no.1 in January, 2012"],
    # 'sub task': ["1.1 Identify store no.1",
    #                 "1.2 Identify item no.9",
    #                 "1.3 Track sales in January, 2012",
    #                 "1.4 Calculate total units sold of item no.9"],
    # 'object': ['units', 'item no', 'store no'],
    # 'implementation': [{'store no.': '1'},
    #                     {'item no.': '9'},
    #                     {'in': 'January, 2012'}]
    # }
    # object 就是 question 里面的关键词。
    # implementation 的字典里面的 value 一般都是一两个词，如果你选的 value 包括很多词的话，再次思考它属不属于 filter 的条件，然后修改。要么是数字，要么是形容词。

    #     '''
    #         }

    #         ]
    #         )
            # keywords=gpt_keyword(task)


        query = f'''
    question: {task.question}
    1. The upper sentence is completely correct. please seperate the upper sentence into main task and sub task.
    2. Tell me how to implement each sub task and divide it into onject and implementation. you can only detect the keywords in the question sentence, do not use words not included in the sentence. 

    ### example one:
    Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?
    Please only respond with a JSON object structured as follows, don't change the keys name:
    {{
    'question':"Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?",
    'task':["1. Name schools in Riverside which the average of average math score for SAT is grater than 400", "2. what is the funding type of these schools?"],
    'sub task':["1.1 find the name of schools in Riverside",
                "1.2 get the average math score of these school", 
                "1.3 calculate the average score of average math score of eah school.", 
                "1.4 find the school which the average of average math score for SAT is grater than 400",
                "2.1 the funding type of these schools"],
    'object':['Name schools'],
    'implementation':[{{'in':'Riverside'}}, {{'the average of':'average math score for SAT'}},{{'is grater than':'400'}},{{'':'funding type'}}]
    }}

    ### example two:
    {{
    'question': "How many units of item no.9 were sold in store no.1 in total in January, 2012?",
    'task': ["Determine the total units sold of item no.9 in store no.1 in January, 2012"],
    'sub task': ["1.1 Identify store no.1",
                    "1.2 Identify item no.9",
                    "1.3 Track sales in January, 2012",
                    "1.4 Calculate total units sold of item no.9"],
    'object': ['units', 'item no', 'store no'],
    'implementation': [{{'store no.': '1'}},
                        {{'item no.': '9'}},
                        {{'in': 'January, 2012'}}]
    }}
    object is related to the keywords in the question.
    the value in the dict of implementation is mostly one to two words，if the values you select contains a lot of word, please double confirm whether it is belong to filter condition, and then revise. it is number or adjective.

    '''

        # keywords, history = inference(model, template, query)

        # print('keywords',query,keywords)
        # keywords=completion.choices[0].message.content
        # print(keywords)
        # 寻找第一个'{'的索引

        # keywords=gpt_keyword(task)
        start_index = keywords.find('{')

        # 寻找最后一个'}'的索引
        end_index = keywords.rfind('}')

        # 提取这两个索引之间的内容，包括大括号本身
        keywords = eval(keywords[start_index:end_index + 1])
        # print("000",keywords)


        ### combine gpt-4o and mistral
        start_index = keywords_2.find('{')

        # 寻找最后一个'}'的索引
        end_index = keywords_2.rfind('}')

        # 提取这两个索引之间的内容，包括大括号本身
        keywords_2 = eval(keywords_2[start_index:end_index + 1])

        keywords['object']=list(set(keywords['object']+keywords_2['object']))
        # print("111",keywords)
        # print(keywords['implementation']+keywords_2['implementation'])
        keywords['implementation']=keywords['implementation']+keywords_2['implementation']
        # print("222",keywords)

        # keywords=keywords_1

            


            # 打印结果
            #print(keywords)
            
            #print(prompt)
            # logging.info("Initiating asynchronous LLM chain call for keyword extraction")
            # response = async_llm_chain_call(
            #     prompt=prompt, 
            #     engine=engine, 
            #     parser=parser,
            #     request_list=[request_kwargs],
            #     step="keyword_extraction",
            #     sampling_count=1
            # )[0]
            # print(task.question)
            # keywords = response[0]
            
            # keywords = {
            # "question": "How many male customers who are living in North Bohemia have average salary greater than 8000?",
            # "task": ["1. Count the male customers living in North Bohemia with an average salary greater than 8000"],
            # "sub task": [
            #     "1.1 identify the male customers",
            #     "1.2 verify the customers living in North Bohemia",
            #     "1.3 determine the average salary of these customers",
            #     "1.4 filter the customers whose average salary is greater than 8000"
            # ],
            # "object": ["male customers","average salary"],
            # "implementation": [
            #     {"living in": "North Bohemia"},
            #     {"greater than ": "8000"}
            # ]
            # }

            # result = {"keywords": eval(keywords)}
            # print('@@@@',keywords)

            # if keywords['question']==task.question:
            #     result = {"keywords": keywords}
            #     break
            # elif i==4:
            #     result = {"keywords": eval(keywords)}
            #     break
            # else:
            #     keywords=gpt_keyword(task)
            #     result = {"keywords": eval(keywords)}
            #     break
        result = {"keywords": keywords}
            # else:
            #     continue
    except:
        for i in range(20):
            try:
                # print(task)
                keywords=gpt_keyword(task)
                # print(keywords)
                # for i in range(5):
                #     if keywords['question']==task.question:
                #         result = {"keywords": eval(keywords)}
                #         break
                #     elif i==4:
                #         result = {"keywords": eval(keywords)}
                #         break
                #     else:
                #         keywords=gpt_keyword(task)
                #         continue
                # print(keywords)
                result = {"keywords": eval(keywords)}
                # print(result)
                if eval(keywords)['question']==task.question:
                    result = {"keywords": eval(keywords)}
                    break
                else:
                    continue
            except:
                continue
    # keywords=gpt_keyword(task)
    # print(result)
    logging.info(f"Keywords extracted: {keywords}")
    return result
