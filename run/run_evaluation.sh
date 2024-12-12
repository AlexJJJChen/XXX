db_root_path='/project/chenjian/nl2sql/CHESS-main/data/dev/dev_databases/'
data_mode='dev'
diff_json_path='/project/chenjian/nl2sql/CHESS-main/data/dev/dev_databases/dev.json'
predicted_sql_path_kg='/project/chenjian/nl2sql/CHESS-main/results/dev/keyword_extraction+entity_retrieval+context_retrieval+candidate_generation+revision+evaluation/debug/2024-09-20-11-21-51/-revision.json'
ground_truth_path='/project/chenjian/nl2sql/CHESS-main/data/dev/dev_databases/dev.sql'
num_cpus=4
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'

python3 -u run/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} 