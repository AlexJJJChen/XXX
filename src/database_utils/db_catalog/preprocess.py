import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch

from database_utils.db_catalog.csv_utils import load_tables_description

load_dotenv(override=True)
# torch.multiprocessing.set_start_method('spawn')
# EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-ada-002")
# EMBEDDING_FUNCTION = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large", azure_endpoint="https://foralexproject.openai.azure.com/", openai_api_key="201a1f2ac52b4d04a0636ce7366ce3ad")
# EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="/project/chenjian/nl2sql/CHESS-main/stella_en_1.5B_v5",multi_process=True)
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
# Initialize the process group
# dist.init_process_group("nccl")


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# print(torch.cuda.is_available())
# print(torch.version.cuda)
# torch.multiprocessing.set_start_method('spawn')
device = torch.device("cuda:3")

from pathlib import Path
EMBEDDINF_ROOT_PATH = Path(os.getenv("EMBEDDING_MODEL"))

EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="/project/chenjian/nl2sql/CHESS-main/stella_en_400M_v5") #, multi_process=True)



def make_db_context_vec_db(db_directory_path: str, **kwargs) -> None:
    """
    Creates a context vector database for the specified database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs: Additional keyword arguments, including:
            - use_value_description (bool): Whether to include value descriptions (default is True).
    """
    print('file name', db_directory_path)
    db_id = Path(db_directory_path).name

    table_description = load_tables_description(db_directory_path, kwargs.get("use_value_description", True))
    docs = []
    
    for table_name, columns in table_description.items():
        for column_name, column_info in columns.items():
            metadata = {
                "table_name": table_name,
                "original_column_name": column_name,
                "column_name": column_info.get('column_name', ''),
                "column_description": column_info.get('column_description', ''),
                "value_description": column_info.get('value_description', '') if kwargs.get("use_value_description", True) else ""
            }
            for key in ['column_name', 'column_description', 'value_description']:
                if column_info.get(key, '').strip():
                    docs.append(Document(page_content=column_info[key], metadata=metadata))
    
    logging.info(f"Creating context vector database for {db_id}")
    vector_db_path = Path(db_directory_path) / "context_vector_db"

    if vector_db_path.exists():
        os.system(f"rm -r {vector_db_path}")

    vector_db_path.mkdir(exist_ok=True)

    Chroma.from_documents(docs, EMBEDDING_FUNCTION, persist_directory=str(vector_db_path))

    logging.info(f"Context vector database created at {vector_db_path}")
