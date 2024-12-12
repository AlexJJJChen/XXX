data_mode='test' # Options: 'dev', 'train' ,'test'

data_path="/project/chenjian/nl2sql/LAIA-SQL/data/test/test.json" #"/project/chenjian/nl2sql/LAIA-SQL/spider_data/test.json" #"/project/chenjian/nl2sql/CHESS-main/data/train/train_mini.json" # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

pipeline_nodes='keyword_extraction+entity_retrieval+context_retrieval+candidate_generation+revision+evaluation'
checkpoint_nodes=''
checkpoint_dir=""

# Nodes:
    # keyword_extraction
    # entity_retrieval
    # context_retrieval
    # column_filtering
    # table_selection
    # column_selection
    # candidate_generation
    # revision
    # evaluation


entity_retieval_mode='ask_model' # Options: 'corrects', 'ask_model'

context_retrieval_mode='vector_db' # Options: 'corrects', 'vector_db'
top_k=2

table_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

column_selection_mode='ask_model' # Options: 'corrects', 'ask_model'

engine1='gemini-pro'
engine2='gpt-3.5-turbo'
engine3='gpt-4-turbo'
engine4='claude-3-opus-20240229'
engine5='gemini-1.5-pro-latest'
engine6='finetuned_nl2sql'
engine7='meta-llama/Meta-Llama-3-70B-Instruct'
engine8='finetuned_colsel'
engine9='finetuned_col_filter'
engine10='gpt-3.5-turbo-instruct'
engine11="gpt-4o-2024-08-06"
engine12="gpt-4o-all"
engine13="gpt-4-turbo-2024-04-09"
engine14="o1-preview-2024-09-12"
engine15="o1-mini-2024-09-12"
engine16="gpt-4o"

pipeline_setup='{
    "keyword_extraction": {
        "engine": "'${engine16}'",
        "temperature": 0.2,
        "base_uri": ""
    },
    "entity_retrieval": {
        "mode": "'${entity_retieval_mode}'"
    },
    "context_retrieval": {
        "mode": "'${context_retrieval_mode}'",
        "top_k": '${top_k}'
    },
    "candidate_generation": {
        "engine": "'${engine16}'",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    },
    "revision": {
        "engine": "'${engine16}'",
        "temperature": 0.0,
        "base_uri": "",
        "sampling_count": 1
    }
}'

echo -e "${run_name}"
python3 -u /project/chenjian/nl2sql/LAIA-SQL/src/main.py --data_mode ${data_mode} --data_path ${data_path}\
        --pipeline_nodes ${pipeline_nodes} --pipeline_setup "$pipeline_setup"\
        # --use_checkpoint --checkpoint_nodes ${checkpoint_nodes} --checkpoint_dir ${checkpoint_dir}
  
