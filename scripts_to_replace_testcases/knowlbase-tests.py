'''
The models are in charge of generating test suites with multiple (up to 10) test cases (methods).
In this script the output of the models containing full suites are processed to created individual suites
where each suite contains only one test method.
'''
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import sys
import ast
from collections import Counter
import javalang

TEST_ANNOTATIONS = ["@Test", "@org.junit.Test"]

def find_block_end(start_line, lines):
    brace_count = 0
    found_start = False

    for i in range(start_line - 1, len(lines)):
        line = lines[i]

        if '{' in line:
            brace_count += line.count('{')
            brace_count -= line.count('}')
            found_start = True
        elif found_start:
            brace_count += line.count('{')
            brace_count -= line.count('}')

        if found_start and brace_count == 0:
            return i + 1

    return start_line


def extract_structure(java_code):
    try:
        tree = javalang.parse.parse(java_code)
    except:
        return []

    lines = java_code.splitlines()

    result = []

    for path, class_node in tree.filter(javalang.tree.ClassDeclaration):
        class_name = class_node.name
        class_start = class_node.position.line if class_node.position else None

        if class_start is None:
            continue

        class_end = find_block_end(class_start, lines)

        class_info = {
            "class_name": class_name,
            "start_line": class_start,
            "end_line": class_end,
            "methods": []
        }

        # Find methods inside this class
        for method in class_node.methods:
            method_name = method.name
            method_start = method.position.line if method.position else None

            if method_start is None:
                continue

            method_end = find_block_end(method_start, lines)

            class_info["methods"].append({
                "name": method_name,
                "start_line": method_start,
                "end_line": method_end
            })

        result.append(class_info)

    return result


def get_javafile_header_stop_line(java_code):
    structure = extract_structure(java_code)
    stop_line = 0
    for class_info in structure:
        stop_line = class_info['start_line']
        class_name = class_info['class_name']
        for method in class_info['methods']:
            if method['name'].lower().startswith("test") and method['name'] != class_name:
                stop_line = method['start_line']
                return stop_line-1
    return stop_line


# file header is everything before the first test method. 
# If there is an annotation before the first test method, exclude it from the header (leave it as part of the method).
# here the header is simply defined as everityhing before the first @Test annotation
def get_javafile_header(java_code):
    if not java_code:
        return None
    
    for annotation in TEST_ANNOTATIONS:
        if annotation in java_code:
            return java_code.split(annotation)[0].rstrip()
    
    return None

# given a java_code and a line number, return the java class containing only the method that starts at that line number.
def class_with_method_at_line(java_file, class_start_line, method_start_line, method_end_line):
    file_header = get_javafile_header(java_file)

    if not file_header:
        return None
    
    method_code = ""
    
    # if the line before the signature starts with an annotation, go up until there are no more annotations
    # include all the annotations in the method code.
    if java_file.splitlines()[method_start_line-2].strip().startswith("@"):
        annotation_lines = []
        up_by_lines = 2
        while java_file.splitlines()[method_start_line-up_by_lines].strip().startswith("@"):
            annotation_lines.append(method_start_line-up_by_lines)
            if method_start_line-up_by_lines == 1:
                break
            up_by_lines += 1
        
        for line_num in reversed(annotation_lines):
            method_code += java_file.splitlines()[line_num] + '\n'
    
    method_code += '\n'.join(java_file.splitlines()[method_start_line-1:method_end_line])
    
    before_closing = file_header + '\n' + method_code

    # count the number of needed closing braces
    needed_closing = before_closing.count('{') - before_closing.count('}')
    closing = '\n}' * needed_closing

    final = before_closing + closing

    return final

# given a java_code, for each method write a file with the class containing only that method. The file name should be the class name + method name + start line number of the method.
def isolate_methods(java_code):
    isolated = []
    method_names = []
    structure = extract_structure(java_code)
    
    if not structure:
        return
    
    for class_info in structure:
        for method in class_info['methods']:
            # keep all the methos, regardless of their name. We will filter them later based on the presence of test annotations.
            isolated.append(class_with_method_at_line(
                java_code,
                class_info['start_line'],
                method['start_line'],
                method['end_line'])
            )
            method_names.append(method['name'])

    return isolated, method_names


def has_closed_strings(s: str) -> bool:
    return s.count("'") % 2 == 0 and s.count('"') % 2 == 0

def has_balanced_brackets(txt: str) -> bool:
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    
    for ch in txt:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack

def looks_incomplete(txt: str) -> bool:
    stripped = txt.rstrip()
    return stripped.endswith((',', '+', '-', '*', '/', '%', 'and', 'or'))

def is_likely_complete(code: str) -> bool:    
    if not has_balanced_brackets(code):
        return False
    if not has_closed_strings(code):
        return False
    if looks_incomplete(code):
        return False
    
    return True


if __name__ == "__main__":

    root_dir = "../test-generations"

    # walk through all subdirectories
    overall_df = pd.DataFrame()
    pattern = r'-JavaBench-\d+\.jsonl$'
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):

            if re.search(pattern, file):
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {os.path.relpath(file_path)}")
                data = []
                with open(file_path, "r") as f:
                    for line in f:
                        data.append(json.loads(line))
                
                df = pd.DataFrame(data)
                overall_df = pd.concat([overall_df, df])
                
    
    overall_df = overall_df[[
        "task_idx", 
        "generated_by",
        "task_id",
        "target",
        "func_name",
        "prompt",
        "raw_output",
    ]].rename(columns={"raw_output": "test_suite"})
    
    num_task_ids = overall_df['task_idx'].nunique()
    
    # explode on outputs
    # extract whats in between the ```java and ``` in test_suite
    overall_df['test_suite'] = overall_df['test_suite'].apply(lambda x: re.search(r'```java(.*?)```', x, re.DOTALL).group(1).strip() if isinstance(x, str) and re.search(r'```java(.*?)```', x, re.DOTALL) else None)
    overall_df[['isolated_tests', 'method_names']] = overall_df['test_suite'].apply(lambda x: pd.Series(isolate_methods(x)) if isinstance(x, str) else pd.Series([None, None]))
    overall_df = overall_df.dropna(subset=['test_suite'])

    # explode on isolated_tests and method_names and remove any Nan
    overall_df = overall_df.explode(['isolated_tests', 'method_names']).reset_index(drop=True)
    overall_df = overall_df.dropna(subset=['isolated_tests', 'method_names'], how='any')

    # exclude tests that don't contain any test annotation. They are probably setup or helper methods.
    overall_df['has_test_annotation'] = overall_df['isolated_tests'].apply(lambda x: any(annotation in x for annotation in TEST_ANNOTATIONS) if isinstance(x, str) else False)
    overall_df = overall_df.loc[overall_df['has_test_annotation'] == True]

    # set test_idx
    overall_df["counter"] = overall_df.groupby(["generated_by", "task_idx"]).cumcount()
    overall_df['test_idx'] = overall_df.apply(lambda row: f"Java--JavaBench--TaskID::{row['task_idx']}--GeneratedBy::{row['generated_by']}--TestID::{row['counter']:02d}", axis=1)

    
    overall_df = overall_df[[
        "task_idx",
        "generated_by",
        "task_id",
        "target",
        "func_name",
        "prompt",
        "isolated_tests",
        "test_idx",
        "method_names",
    ]].rename(columns={'isolated_tests': 'test_statement', 'method_names': 'test_method_name'})

    # CLEANING
    overall_df = overall_df.loc[~overall_df['test_statement'].isin([""])]
    overall_df['is_likely_complete'] = overall_df['test_statement'].apply(lambda x: is_likely_complete(x))
    overall_df = overall_df.loc[overall_df['is_likely_complete']].drop(columns=['is_likely_complete'])
    
    # no more than XX unique asserts per task ID per model
    overall_df['task_idx_genby'] = overall_df['test_idx'].apply(lambda x: x.split('--TestID::')[0])
    overall_df['counter'] = overall_df.groupby('task_idx_genby').cumcount()
    overall_df = overall_df[overall_df['counter'] < 10].drop(columns=['task_idx_genby', 'counter'])
    # END CLEANING

    g = overall_df.groupby(['generated_by', 'task_idx'], as_index=False).count()
    print()
    
    # authorize saving form input
    
    for generated_by in g['generated_by'].unique():
        print(f"\n################## {generated_by} ##################")
        print(f"Max number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].max()}")
        print(f"Min number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].min()}")
        print(f"Average number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].mean():.2f}")
        print(f"Median number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].median()}")
        print(f"Number of task IDs with no asserts (out of {num_task_ids}):\t\t{num_task_ids - g[g['generated_by'] == generated_by]['task_idx'].nunique()}")
        print()
    
        save_dir = "../data/knowlbase-tests"
        authorized = input(f"Do you want to save the tests extracted for {generated_by} to '{save_dir}'? (y/n): ")
        if authorized.lower() in ['y', 'yes']:
            save_path = os.path.join(save_dir, f"{generated_by}_knowlbase_tests_javabench.jsonl")
            genby = overall_df[overall_df['generated_by'] == generated_by]
            genby.to_json(save_path, orient='records', lines=True)
            print(f"\nSaved overall extracted tests to '{save_path}'")
        else:
            print("Skipped.")