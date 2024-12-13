import numpy as np
import difflib
import concurrent.futures
import logging
from typing import List, Dict, Tuple, Optional, Any

from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings,HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

import os

from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
# from database_utils.db_catalog.preprocess import EMBEDDING_FUNCTION




os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


device = torch.device("cuda:3")

from pathlib import Path
EMBEDDINF_ROOT_PATH = os.getenv("EMBEDDING_MODEL")
# 加载模型并指定单个 GPU
EMBEDDING_FUNCTION = SentenceTransformer(EMBEDDINF_ROOT_PATH, trust_remote_code=True).cuda()

@node_decorator(check_schema_status=False)
def entity_retrieval(task: Any, tentative_schema: Dict[str, Any], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Retrieves entities and columns similar to given keywords from the task.

    Args:
        task (Any): The task object containing the evidence and question.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing similar columns and values.
    """
    logging.info("Starting entity retrieval")
    keywords = get_last_node_result(execution_history, "keyword_extraction")["keywords"]
    #print('keywords',keywords)
    for i in range(1):
        try:
            #.extend([list(d.keys())[0] for d in keywords['implementation']])
            similar_columns = get_similar_columns(keywords=keywords['object'], question=task.question, hint=task.evidence)
            # print('similar_columns 01',similar_columns)
            result = {"similar_columns": similar_columns}
            # print('similar_columns:',result['similar_columns'])
            # print(list(set([list(d.values())[0] for d in keywords['implementation']])))
    
            similar_values = get_similar_entities(keywords=list(set([list(d.values())[0] for d in keywords['implementation']])),target_column=result['similar_columns'])
            # print("similar_values",similar_values)
            result["similar_values"] = similar_values
            break
        except Exception as e:
            print(e)
            continue
    
    #print('similar_values:',result)

    logging.info("Entity retrieval completed successfully")
    return result

### Column name similarity ###

def get_similar_columns(keywords: List[str], question: str, hint: str) -> Dict[str, List[str]]:
    """
    Finds columns similar to given keywords based on question and hint.

    Args:
        keywords (List[str]): The list of keywords.
        question (str): The question string.
        hint (str): The hint string.

    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of similar column names.
    """
    logging.info("Retrieving similar columns")
    selected_columns = {}
    for keyword in keywords:
        #print(keyword)
        similar_columns = _get_similar_column_names(keyword=keyword, question=question, hint=hint)
        #print('similar_columns 02', similar_columns)
        for table_name, column_name in similar_columns:
            selected_columns.setdefault(table_name, []).append(column_name)
            #print('selected_columns', selected_columns)
            
    return selected_columns

def _column_value(string: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Splits a string into column and value parts if it contains '='.

    Args:
        string (str): The string to split.

    Returns:
        Tuple[Optional[str], Optional[str]]: The column and value parts.
    """
    if "=" in string:
        left_equal = string.find("=")
        first_part = string[:left_equal].strip()
        second_part = string[left_equal + 1:].strip() if len(string) > left_equal + 1 else None
        return first_part, second_part
    return None, None

def _extract_paranthesis(string: str) -> List[str]:
    """
    Extracts strings within parentheses from a given string.

    Args:
        string (str): The string to extract from.

    Returns:
        List[str]: A list of strings within parentheses. transfer words to alphabets.
    """
    paranthesis_matches = []
    open_paranthesis = []
    for i, char in enumerate(string):
        #print(char)
        if char == "(":
            open_paranthesis.append(i)
        elif char == ")" and open_paranthesis:
            start = open_paranthesis.pop()
            found_string = string[start:i + 1]
            if found_string:
                paranthesis_matches.append(found_string)
    return paranthesis_matches

def _does_keyword_match_column(keyword: str, column_name: str, threshold: float = 0.7) -> bool: # default is 0.7
    """
    Checks if a keyword matches a column name based on similarity.

    Args:
        keyword (str): The keyword to match.
        column_name (str): The column name to match against.
        threshold (float, optional): The similarity threshold. Defaults to 0.9.

    Returns:
        bool: True if the keyword matches the column name, False otherwise.
    """
    keyword = keyword.lower().replace(" ", "").replace("_", "").rstrip("s")
    column_name = column_name.lower().replace(" ", "").replace("_", "").rstrip("s")
    similarity = difflib.SequenceMatcher(None, column_name, keyword).ratio()
    #print('similarity',similarity)
    return similarity >= threshold

def _get_similar_column_names(keyword: str, question: str, hint: str) -> List[Tuple[str, str]]:
    """
    Finds column names similar to a keyword.

    Args:
        keyword (str): The keyword to find similar columns for.
        question (str): The question string.
        hint (str): The hint string.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing table and column names.
    """
    keyword = keyword.strip()
    potential_column_names = [keyword]
    #print(potential_column_names)

    column, value = _column_value(keyword)
    if column:
        potential_column_names.append(column)

    potential_column_names.extend(_extract_paranthesis(keyword))

    if " " in keyword:
        potential_column_names.extend(part.strip() for part in keyword.split())

    schema = DatabaseManager().get_db_schema()
    # print('schema',schema)

    similar_column_names = []
    for table, columns in schema.items():
        for column in columns:
            for potential_column_name in potential_column_names:
                if _does_keyword_match_column(potential_column_name, column):
                    similarity_score = _get_semantic_similarity_with_openai(f"`{table}`.`{column}`", [f"{question} {hint}"])[0]
                    similar_column_names.append((table, column, similarity_score))

    similar_column_names.sort(key=lambda x: x[2], reverse=True)
    #print('')
    return [(table, column) for table, column, _ in similar_column_names[:1]]

### Entity similarity ###

def get_similar_entities(keywords: List[str], target_column:{}) -> Dict[str, Dict[str, List[str]]]:
    """
    Retrieves similar entities from the database based on keywords.

    Args:
        keywords (List[str]): The list of keywords.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column names to similar entities.
    """
    logging.info("Retrieving similar entities")
    selected_values = {}

    def get_similar_values_target_string(target_string: str):
        unique_similar_values_list = DatabaseManager().query_lsh(keyword=target_string, signature_size=100, top_n=10)
        if not isinstance(unique_similar_values_list, dict):
            # print("unique_similar_values_list 不是字典类型")
            return target_string, None
        
        unique_similar_values = {}
        # print('unique_similar_values_list', unique_similar_values_list)
        # print('key_list',list(unique_similar_values_list.keys()))
        # print('target_column',list(target_column.keys()))
        
        for key in list(unique_similar_values_list.keys()):
            if key in list(target_column.keys()):  # 确保target_column已经被定义
                # print('key', key)
                unique_similar_values[key] = unique_similar_values_list[key]
                # print('unique_similar_values', unique_similar_values)

        return target_string, _get_similar_entities_to_keyword(target_string, unique_similar_values)

    for keyword in keywords:
        try:
            keyword = keyword.strip()
            to_search_values = [keyword]
            if (" " in keyword) and ("=" not in keyword):
                for i in range(len(keyword)):
                    if keyword[i] == " ":
                        first_part = keyword[:i]
                        second_part = keyword[i+1:]
                        if first_part not in to_search_values:
                            to_search_values.append(first_part)
                        if second_part not in to_search_values:
                            to_search_values.append(second_part)

            to_search_values.sort(key=len, reverse=True)
            hint_column, hint_value = _column_value(keyword)
            if hint_value:
                to_search_values = [hint_value, *to_search_values]
            

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(get_similar_values_target_string, ts): ts for ts in to_search_values}
                # print('futures',futures)
                for future in concurrent.futures.as_completed(futures):
                    target_string, similar_values = future.result()
                    for table_name, column_values in similar_values.items():
                        for column_name, entities in column_values.items():
                            # print("column_name",column_name)
                            if entities:
                                # print("entities",entities)
                                try:
                                    # print("selected_values_1",selected_values)
                                    selected_values.setdefault(table_name, {}).setdefault(column_name, []).extend(
                                        [(ts, value, edit_distance, embedding) for ts, value, edit_distance, embedding in entities]
                                    )
                                    # print("selected_values_2",selected_values)
                                except:
                                    print('222222',entities)
                                    continue
                            # print(selected_values)
        except:
            continue
    # print('!!!!!!!!')
    for table_name, column_values in selected_values.items():
        # print('^^^^^^^^')
        
        for column_name, values in column_values.items():
            max_edit_distance = max(values, key=lambda x: x[2])[2]
            # print(max_edit_distance)
            try:
                # print("values",values,max_edit_distance)
                selected_values[table_name][column_name] = list(set(
                    value for _, value, edit_distance, __ in values if edit_distance == max_edit_distance
                ))
                # print('@@@@@@@@@')
            except:
                # print('#########')
                continue
            # list index out of range
            # print('selected_values',selected_values)
    return selected_values

def _get_similar_entities_to_keyword(keyword: str, unique_values: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[Tuple[str, str, float, float]]]]:
    """
    Finds entities similar to a keyword in the database.

    Args:
        keyword (str): The keyword to find similar entities for.
        unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values from the database.

    Returns:
        Dict[str, Dict[str, List[Tuple[str, str, float, float]]]]: A dictionary mapping table and column names to similar entities.
    """
    return {
        table_name: {
            column_name: _get_similar_values(keyword, values)
            for column_name, values in column_values.items()
        }
        for table_name, column_values in unique_values.items()
    }

def _get_similar_values(target_string: str, values: List[str]) -> List[Tuple[str, str, float, float]]:
    """
    Finds values similar to the target string based on edit distance and embedding similarity.

    Args:
        target_string (str): The target string to compare against.
        values (List[str]): The list of values to compare.

    Returns:
        List[Tuple[str, str, float, float]]: A list of tuples containing the target string, value, edit distance, and embedding similarity.
    """
    edit_distance_threshold = 0.3
    top_k_edit_distance = 5

    if target_string.isdigit():
        embedding_similarity_threshold = 1
    elif target_string.isalpha():
        embedding_similarity_threshold = 0
    else:
        embedding_similarity_threshold = 0
    top_k_embedding = 1

    edit_distance_similar_values = [
        (value, difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio())
        for value in values
        if difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio() >= edit_distance_threshold
    ]
    # print('edit_distance_similar_values',edit_distance_similar_values)
    edit_distance_similar_values.sort(key=lambda x: x[1], reverse=True)
    edit_distance_similar_values = edit_distance_similar_values[:top_k_edit_distance]
    # print('edit_distance_similar_values',edit_distance_similar_values)
    # print([value for value, _ in edit_distance_similar_values])
    similarities = _get_semantic_similarity_with_openai(target_string, [value for value, _ in edit_distance_similar_values])
    # print('similarities',target_string,similarities)

    if target_string.isalpha():
        embedding_similar_values = [
            (target_string, edit_distance_similar_values[edit_distance_similar_values.index(max(edit_distance_similar_values))][0], edit_distance_similar_values[edit_distance_similar_values.index(max(edit_distance_similar_values))][1], similarities[edit_distance_similar_values.index(max(edit_distance_similar_values))])
            # for i in range(len(edit_distance_similar_values))
            # if similarities[i] >= embedding_similarity_threshold
        ]
    else:
        embedding_similar_values = [
            (target_string, edit_distance_similar_values[i][0], edit_distance_similar_values[i][1], similarities[i])
            for i in range(len(edit_distance_similar_values))
            if similarities[i] >= embedding_similarity_threshold
        ]

    embedding_similar_values.sort(key=lambda x: x[2], reverse=True)
    # print('embedding_similar_values',embedding_similar_values)
    return embedding_similar_values[:top_k_embedding]

def _get_semantic_similarity_with_openai(target_string: str, list_of_similar_words: List[str]) -> List[float]:
    """
    Computes semantic similarity between a target string and a list of similar words using OpenAI embeddings.

    Args:
        target_string (str): The target string to compare.
        list_of_similar_words (List[str]): The list of similar words to compare against.

    Returns:
        List[float]: A list of similarity scores.
    """
    #print("11111111########",target_string)
    try:
        target_string_embedding = EMBEDDING_FUNCTION.embed_query(target_string)
        # print('target_string',target_string)
        all_embeddings = EMBEDDING_FUNCTION.embed_documents(list_of_similar_words)
        similarities =  [np.dot(target_string_embedding, embedding) for embedding in all_embeddings]

    except:

        # use scl embedding model
        target_string_embedding = EMBEDDING_FUNCTION.encode(list(target_string), prompt_name='s2s_query')
        all_embeddings = EMBEDDING_FUNCTION.encode(list_of_similar_words)
        similarities =  EMBEDDING_FUNCTION.similarity(target_string_embedding, all_embeddings)[0].tolist() #[np.dot(target_string_embedding, embedding) for embedding in all_embeddings]

    # print(target_string)
    # print('list_of_similar_words',list_of_similar_words)
    # print(similarities)
    return similarities
