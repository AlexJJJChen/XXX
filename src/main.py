import argparse
import json
from datetime import datetime
from typing import Any, Dict, List

from runner.run_manager import RunManager
import torch
import os
import time
import torch.multiprocessing as mp
# print(torch.cuda.is_available())
# print(torch.version.cuda)
torch.cuda.empty_cache()
#print(torch.cuda.memory_summary())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES']='3'
def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--pipeline_nodes', type=str, required=True, help="Pipeline nodes configuration.")
    parser.add_argument('--pipeline_setup', type=str, required=True, help="Pipeline setup in JSON format.")
    parser.add_argument('--use_checkpoint', action='store_true', help="Flag to use checkpointing.")
    parser.add_argument('--checkpoint_nodes', type=str, required=False, help="Checkpoint nodes configuration.")
    parser.add_argument('--checkpoint_dir', type=str, required=False, help="Directory for checkpoints.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    args = parser.parse_args()

    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    
    if args.use_checkpoint:
        print('Using checkpoint')
        if not args.checkpoint_nodes:
            raise ValueError('Please provide the checkpoint nodes to use checkpoint')
        if not args.checkpoint_dir:
            raise ValueError('Please provide the checkpoint path to use checkpoint')

    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    #print(dataset)
    return dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    start_time = time.time()  # 记录开始时间
    args = parse_arguments()
    #  print(args)
    dataset = load_dataset(args.data_path)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

    end_time = time.time()  # 记录结束时间
    print(f"运行时间：{end_time - start_time:.2f}秒")

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    mp.set_start_method('spawn', force=True) 
    main()
