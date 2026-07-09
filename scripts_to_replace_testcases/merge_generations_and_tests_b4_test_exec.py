'''
This script takes as input the knowledge base of the code solutions and
the knowledge base of the test generations, and merges them to create a new knowledge base 
where each entry is a <code_solution, test_statement> pair. 
'''
import os
import re
import json
import pandas as pd
import numpy as np
import sys
import gzip
from tqdm import tqdm
from collections import Counter

if __name__ == "__main__":
    tqdm.pandas()
    
    # IMPORT CODE KNOWLEDGE BASE
    kb_dir = "../../JavaBench/knowlbase"
    kb = pd.DataFrame()
    for file in os.listdir(kb_dir):
        if re.search(r'knowlbase_javabench_part\d+\.json', file):
            print(f"Processing file: {file}")
            knowlbase_path = os.path.join(kb_dir, file)
            with open(knowlbase_path, "r") as f:
                kb_j = json.load(f)
            kb_t = pd.DataFrame(kb_j)
            kb = pd.concat([kb, kb_t], ignore_index=True)

    kb = kb[[
        'solution_idx',
        'task_idx',
        'generated_by',
        'prompt',
        'method',
        "is_pass",
        
        'additional_info',
        ]]
    
    kb['completion'] = kb['additional_info'].apply(lambda x: x.get('completion', '') if isinstance(x, dict) else '')
    kb['onwhichtomerge'] = kb['solution_idx'].apply(lambda x: x.split("--SampleID")[0])

    # FILTER KNOWLEDGE BASE TO TEST SET SOLUTIONS
    with open("../data/finetuning/test_upto20_real.jsonl", "r") as f:
        test_set_uptp20 = [json.loads(line) for line in f]
    test_set_uptp20 = pd.DataFrame(test_set_uptp20)
    solution_ids_to_keep = test_set_uptp20.solution_idx.unique().tolist()
    kb = kb[kb.solution_idx.isin(solution_ids_to_keep)]
    del test_set_uptp20
    del solution_ids_to_keep
    
    
    # SPLIT BY GENERATED_BY
    for generated_by in kb['generated_by'].unique():

        # IMPORT TESTS KNOWLEDGE BASE
        test_path = f"../knowlbase-tests/{generated_by}_knowlbase_tests_javabench.jsonl"
        if not os.path.exists(test_path):
            print(f"Test knowledge base file '{test_path}' not found. Skipping '{generated_by}'.")
            continue
        kbt_genby = pd.read_json(test_path, lines=True).drop(columns=['prompt'])
        kbt_genby['onwhichtomerge'] = kbt_genby['test_idx'].apply(lambda x: x.split("--TestID")[0])
        kb_genby = kb.loc[kb['generated_by'] == generated_by]
        kbt_genby = kbt_genby.drop(columns=['task_idx', 'generated_by'])

        
        merged = kb_genby.merge(kbt_genby, on="onwhichtomerge", how="inner")
        
        ##################
        g = merged.groupby('solution_idx', as_index=False).agg({'test_idx': 'count', 'task_idx': 'first'})
        num_task_ids = json.load(open('../../../constants/ids_train_val_test.json', 'r'))['JavaBench']['test']
        print(f"################## {generated_by} ##################")
        print(kb_genby.task_idx.nunique())
        print(f"Max number of unique asserts per task ID:\t\t{g['test_idx'].max()}")
        print(f"Min number of unique asserts per task ID:\t\t{g['test_idx'].min()}")
        print(f"Average number of unique asserts per task ID:\t\t{g['test_idx'].mean():.2f}")
        print(f"Median number of unique asserts per task ID:\t\t{g['test_idx'].median()}")
        print(f"Number of task IDs with no asserts (out of {len(num_task_ids)}):\t\t{len(num_task_ids) - g['task_idx'].nunique()}")
        print()
        ##################

        # cast the patch column to list because the test execution framework of ujb expects a list
        merged['patch'] = merged['method'].apply(lambda x: [x])
        
        # unique id of the pair <code_solution, test_statement>
        merged['test_execution_idx'] = merged.apply(lambda row : \
        '{}--TestID::{}--SampleID::{}'.format(
                row['solution_idx'].split('--SampleID::')[0],
                row['test_idx'].split('--TestID::')[-1],
                row['solution_idx'].split('--SampleID::')[-1]
            ), axis=1
        )

        merged = merged[[
            "task_idx",
            "generated_by",
            "prompt",
            "task_id",
            "target",
            "func_name",
            "test_idx",
            "test_statement",
            "test_method_name",
            "completion",
            "method",
            "test_execution_idx"
        ]]
        
        # authorize saving form input
        save_dir = "../test-4-execution"
        authorized = input(f"Do you want to save the tests extracted for {generated_by} to '{save_dir}'? (y/n): ")
        if authorized.lower() in ['y', 'yes']:
            # chunk merged and save to json
            chunk_size = 1000
            num_chunks = (len(merged) + chunk_size - 1) // chunk_size
            print(f"Saving merged data to {save_dir} in {num_chunks} chunks...")
            for i in tqdm(range(num_chunks)):
                chunk = merged.iloc[i*chunk_size:(i+1)*chunk_size]
                chunk.to_json(os.path.join(save_dir, f"{generated_by}_test4execution_{i:02d}.jsonl"), lines=True, orient='records')
        else:
            print("Skipped.")
