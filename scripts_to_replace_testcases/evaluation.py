# ./evaluation.py
import json
import json.tool
import os
import sys
import click
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import networkx as nx

from app.test_env import TestEnv
from app.util.io import extract_code, stream_jsonl

def evaluate_test_suite(
    sample_file: str,
    output: str,
    *,
    mode: str = "full",
    test_file: str = "data/test.jsonl",
):
    samples = stream_jsonl(sample_file)
    grouped_samples = {}
    for sample in samples:
        target = sample["target"].rsplit(".")[0].replace("/", ".")
        if not grouped_samples.get(target):
            grouped_samples[target] = []
        grouped_samples[target].append(sample)

    tests = list(stream_jsonl(test_file))
    test_results = {}
    for test_index, test in enumerate(tests):
        if mode == "inc" and len(test["incremental_deps"]) == 0:
            continue
        if mode == "full":
            target_samples = [grouped_samples[dep] for dep in test["full_deps"]]
        elif mode == "inc":
            target_samples = [grouped_samples[dep] for dep in test["incremental_deps"]]
        else:
            raise ValueError(f"Unknown mode {mode}")
        
        if len(target_samples) == 0:
            continue

        result = []
        project_id, test_id = test["test_id"].split("/")
        max_k = min(len(l) for l in grouped_samples.values())
        test_env = TestEnv(
            root=f"/tmp/pre-coder/{os.getpid()}-{mode}/{project_id}-{test_id}",
            todo_src=f"projects/{project_id}",
            src=f"projects/{project_id}-Solution",
        )
        for k in range(max_k):
            print(f"[{os.getpid()}/{mode}] Running test {test_index + 1}/{len(tests)}: {test['test_id']} k={k + 1}")
            samples_to_replace = [target_sample[k] for target_sample in target_samples]

            has_todo = False
            can_replace = True
            for sample in samples_to_replace:
                code = extract_code(sample["completion"])
                replace_result = test_env.replace(sample["target"], code)

                has_todo = has_todo or replace_result["has_todo"]
                can_replace = can_replace and replace_result["can_replace"]

            compilable = len(test_env.compile()) == 0
            if compilable:
                (n_pass, n_total), _ = test_env.run_test(test["target"])
            else:
                n_pass, n_total = 0, 0

            result.append(
                dict(
                    test_id=test["test_id"],
                    compilable=compilable,
                    n_pass=[n_pass, n_total],
                    has_todo=has_todo,
                    can_replace=can_replace,
                )
            )
        test_results[test["test_id"]] = result
        test_env.destroy()


    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as fp:
        fp.write((json.dumps(test_results, indent=4) + "\n"))

@click.command()
@click.argument("data")
@click.option('--output', required=True, help="Output file for evalution")
@click.option('--test', required=True, help="Test configuration for evaluation")
def test_wise(data: str, output: str, test: str):
    evaluate_test_suite(data, output, test_file=test, mode="full")


def rdm_string_generator(length):
    import random
    import string
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def execution_errors_from_stdout(stdout: str):
    """Example of the pattern 1 to match:
    MapTest > givenSubsequentPipe_ifCanFillPipeFromCorrectDirection_thenSuccess() FAILED
        org.opentest4j.AssertionFailedError at MapTest.java:87
    
    Example of the pattern 2 to match:
    GameControllerTest > Make Move - Move to Adjacent Border, Unlimited Lives > pa1.controller.GameControllerTest.testMakeMoveToBorderUnlimitedLives(Direction)[1] FAILED
        java.lang.NullPointerException at GameControllerTest.java:111
    
    Example of the pattern 3 to match:
    GameBoardControllerTest > Make Move - Move to Wall FAILED
        org.opentest4j.AssertionFailedError at GameBoardControllerTest.java:193
    """
    patterns_to_try = [
        r"(\w+) > (\w+)\(\) FAILED\s+[\w\.]+ at (\w+\.java):\d+",
        r"(\w+) > [\w\.\s\-\,]+ > ([\w\.]+)[\(\)\w\,\[\]\d]+ FAILED\s+[\w\.]+ at (\w+\.java):\d+",
        r"(\w+) > ([\-\w\,\s]+) FAILED\s+[\w\.]+ at (\w+\.java):\d+"
    ]
    test_error_from_stdout = []
    test_error_from_stdout_idx = []
    test_class_counter = {}
    for ipattern, pattern in enumerate(patterns_to_try):
        matches = re.findall(pattern, stdout)
        for match in matches:
            test_class, test_method_name, test_file_name = match
            if ipattern != len(patterns_to_try) - 1: # if not last pattern include method name
                if '.' in test_method_name:
                    test_method_name = test_method_name.split('.')[-1]
                test_error_from_stdout.append((test_class, test_method_name, test_file_name))
            else:
                test_error_from_stdout.append((test_class, None, test_file_name))
            test_class_counter.setdefault(test_class, 0)
            test_error_from_stdout_idx.append(f"{test_class_counter[test_class]}{test_class}")
            test_class_counter[test_class] += 1
        
        if len(test_error_from_stdout) > 0:
            d = dict(zip(test_error_from_stdout_idx, test_error_from_stdout))
            return d, f'pattern{ipattern + 1}'
    return {}, '-'
          
def parse_execution_log(
    execution_log: str,
    test_method_name: str,
    test_file_name: str,
    root: str,
    project_id: str,
    file_name: str,
    method_name: str,

    error_line_number: list,
    failed_test_expected_output: list,
    failed_test_actual_output: list,
    error_message: list,
    error_type: list,
    error_line_code: list,
    exec_feedback: list,
):
    eln, fteo, ftao, em, et, elc, ef, rel_path_to_test_file = -1, "-", "-", "-", "-", "-", "-", "-"
    # now that we have the error log, extract the line number and file name of the error
    for line in execution_log.splitlines():
        # look for a like containing the test method name and test file name
        # e.g., ...
        if test_method_name in line and test_file_name in line:
            line = line.replace('lambda$', '')
            pattern = r'\.java:(\d+)'
            match = re.search(pattern, line)
            if match:
                eln = int(match.group(1))
                et = execution_log.splitlines()[0].split(':')[0].strip()
                em = execution_log.splitlines()[0].split(f"{et}:")[-1].strip()
                
                # check if the error message contains expected and actual output
                if 'expected: ' in em and ' but was: ' in em:
                    fteo = em.split('expected: ')[-1].split(' but was: ')[0].strip()
                    ftao = em.split(' but was: ')[-1].strip()
                elif 'expected: not <null>' in em:
                    fteo = 'not <null>'
                    ftao = 'null'
            
                rel_path_to_test_file = line.split('at app//')[-1].split(f'.{test_method_name}')[0] \
                    if line.strip().startswith('at app//') else line.split('at ')[-1].split(f'.{test_method_name}')[0]
                rel_path_to_test_file = rel_path_to_test_file.replace('.', '/') + '.java'
                
                # read the test file to get the code line that caused the error
                try:
                    with open(os.path.join(root, 'src', 'test', 'java', rel_path_to_test_file), 'r') as r:
                        test_file = r.readlines()
                    
                    elc = test_file[eln - 1].strip()
                    ef = f"Test Failed--\nFile: {rel_path_to_test_file}\nLine: {eln}\nMessage: {em}\nContent:\n{elc}\n"
                except Exception as e:
                    ef = f"Test Failed--\nFile: {rel_path_to_test_file}\nLine: {eln}\nMessage: {em}\nContent:\n{elc}\n"
                    

                error_line_number.append(eln)
                failed_test_expected_output.append(fteo)
                failed_test_actual_output.append(ftao)
                error_message.append(em)
                error_type.append(et)
                error_line_code.append(elc)
                exec_feedback.append(ef)

                return

def soup_parser_from_html_report(html_report_path, tag):
    rel_path = tag['href'].split('#')[0]
    test_method_name = tag['href'].split('#')[-1].strip("()")
    html_log_path = '/'.join(html_report_path.split('/')[:-1] + [rel_path])
    with open(html_log_path, 'r') as file:
        html = file.read()
    return BeautifulSoup(html, 'html.parser'), test_method_name

def evaluate_single_class(
    sample_file: str,
    output: str,
):
    result = []
    samples = list(stream_jsonl(sample_file))
    taskid_testfile_map = json.load(open("./constants/taskid_testfilepath.json", "r"))
    for sample in tqdm(samples):
        project_id = sample["task_id"].split("/")[0]
        file_name = sample["target"].split("/")[-1].split(".java")[0]
        method_name = sample["func_name"]
        custom_test_method_name = sample["test_method_name"]
        test_statement = sample["test_statement"]
        rdm_string = rdm_string_generator(16)
        root = f"/tmp/pre-coder/{rdm_string}-single_class/{sample['task_id'].rsplit('.')[0]}"
        test_env = TestEnv(
            root=root,
            todo_src=f"projects/{project_id}",
            src=f"projects/{project_id}-Solution",
        )
        replace_result = test_env.replace(sample["target"], sample["completion"])

        # errors = test_env.compile()
        # result.append(
        #     dict(
        #         task_id=sample["task_id"],
        #         compile_errors_count=len(errors),
        #         compile_errors=[
        #             {
        #                 "source": error.source,
        #                 "line": error.line,
        #                 "message": error.message,
        #                 "content": error.content,
        #             } for error in errors
        #         ],
        #     )
        # )

        # default values for the 7 error info fields
        error_line_number = []
        failed_test_input = []
        failed_test_expected_output = []
        failed_test_actual_output = []
        error_message = []
        error_type = []
        error_line_code = []
        exec_feedback = []
        full_log = []
        test_result = None

        # if something in the generation went wrong, don't execute the tests
        if sample["completion"] != "GENERATION_OR_EXTRACTION_FAILED":
        
            # run tests
            compile_errors = test_env.compile()
            compile_result = len(compile_errors)
            
            # in case of compilation errors
            if compile_result > 0:
                is_pass = False
                for error in compile_errors:
                    error_line_number.append(error.line)
                    error_message.append(error.message)
                    error_type.append("SyntaxError")
                    error_line_code.append(error.content.split('\n')[0].strip())
                    exec_feedback.append(f"Compilation error--\nFile: {error.source}\nLine: {error.line}\nMessage: {error.message}\nContent:\n{error.content}\n")
                    
            # if compilation is successful, run the tests
            else:
                last_part_path = taskid_testfile_map[sample["task_idx"]]
                test_result, stdout, stderr = test_env.run_test_tests(
                    test_statement=test_statement,
                    last_part_path=last_part_path,
                    custom_test_method_name=custom_test_method_name,  
                    )
                is_pass = test_result[0] == test_result[1] and test_result[1] > 0

                # execution feedback extraction
                if replace_result["can_replace"] and not is_pass:
                    
                    # if the stderr is not empty, it means that some execution error happened
                    for line in stderr.splitlines():
                        # look for the html file containing the test report
                        if "> There were failing tests. See the report at: file:///private" in line:
                            html_report_path = line.split("file:///private")[-1].strip()
                            with open(html_report_path, 'r') as r:
                                report = r.read()

                            execution_errors, patt = execution_errors_from_stdout(stdout)

                            for idx, (test_class, test_method_name, test_file_name) in execution_errors.items():
                                soup_report = BeautifulSoup(report, 'html.parser')
                                if patt == 'pattern1':
                                    tag = soup_report.find("a", string=lambda text: text and test_method_name in text)
                                    soup, _ = soup_parser_from_html_report(html_report_path, tag) # open the file actually containing the error log
                                    tag = soup.find("h3", string=lambda text: text and test_method_name in text)

                                elif patt == 'pattern2':
                                    tag = soup_report.find("a", string=test_class).find_next_sibling("a")
                                    soup, _ = soup_parser_from_html_report(html_report_path, tag)
                                    tag = soup.find("a", attrs={"name": lambda t: t and test_method_name in t})
                                
                                elif patt == 'pattern3':
                                    tags = soup_report.find_all("a", string=test_class)
                                    idx = int(idx[:-len(test_class)])
                                    tag = tags[idx].find_next_sibling("a")
                                    soup, test_method_name = soup_parser_from_html_report(html_report_path, tag)
                                    tag = soup.find("a", attrs={"name": lambda t: t and test_method_name in t})
                                    

                                
                                tag = tag.find_next_sibling("span", class_="code")
                                execution_log = tag.text.strip()
                                full_log.append(execution_log)
                                parse_execution_log(
                                                    execution_log,
                                                    test_method_name,
                                                    test_file_name,
                                                    root,
                                                    project_id,
                                                    file_name,
                                                    method_name,

                                                    error_line_number,
                                                    failed_test_expected_output,
                                                    failed_test_actual_output,
                                                    error_message,
                                                    error_type,
                                                    error_line_code,
                                                    exec_feedback,
                                                )
        
        result.append(
            dict(
                test_execution_idx=sample["test_execution_idx"],
                task_idx=sample["task_idx"],
                sample_idx=sample["sample_idx"],
                generated_by=sample["generated_by"],
                prompt=sample["prompt"],
                method=sample["method"],
                completion=sample["completion"],
                test_idx=sample["test_idx"],
                test_statement=test_statement,
                # raw_output=sample["raw_output"],
                # location=sample["target"],
                is_pass=is_pass,
                compile_errors=compile_result,
                exec_feedback=str(exec_feedback),
                error_line_number=str(error_line_number),
                failed_test_input=str(failed_test_input),
                failed_test_expected_output=str(failed_test_expected_output),
                failed_test_actual_output=str(failed_test_actual_output),
                error_message=str(error_message),
                error_type=str(error_type),
                error_line_code=str(error_line_code),

                test_result=test_result or [0, 0],
                full_log=str(full_log),
                # has_todo=replace_result["has_todo"],
                # can_replace=replace_result["can_replace"],
            )
        )
        test_env.destroy()
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as fp:
        fp.write((json.dumps(result, indent=4) + "\n"))


@click.command()
@click.argument("data")
@click.option('--output', required=True, help="Output file for evalution")
def class_wise(data: str, output: str):
    evaluate_single_class(data, output)

@click.command()
@click.argument("data")
@click.option('--output', required=True, help="Output file for evalution")
def project_wise(data: str, output: str):
    samples = stream_jsonl(data)
    grouped_samples = {}
    for sample in samples:
        target = sample["target"].rsplit(".")[0].replace("/", ".")
        if not grouped_samples.get(target):
            grouped_samples[target] = []
        grouped_samples[target].append(sample)

    project_id = sample["task_id"].split("/")[0]
    result = []
    max_k = min(len(l) for l in grouped_samples.values())
    test_env = TestEnv(
        root=f"/tmp/pre-coder/{os.getpid()}-project",
        todo_src=f"projects/{project_id}",
        src=f"projects/{project_id}-Solution",
    )
    for k in range(max_k):
        print(f"[{os.getpid()}/project] Running project {k + 1}...")
        samples_to_replace = [target_sample[k] for target_sample in grouped_samples.values()]

        has_todo = False
        can_replace = True
        for sample in samples_to_replace:
            code = extract_code(sample["completion"])
            replace_result = test_env.replace(sample["target"], code)

            has_todo = has_todo or replace_result["has_todo"]
            can_replace = can_replace and replace_result["can_replace"]

        compile_error = test_env.compile()
        if len(compile_error) == 0:
            (n_pass, n_total), log = test_env.run_test(None)
        else:
            n_pass, n_total = 0, 0

        result.append(
            dict(
                compile_error=len(compile_error),
                n_pass=[n_pass, n_total],
                has_todo=has_todo,
                can_replace=can_replace,
            )
        )
    test_env.destroy()

    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as fp:
        fp.write((json.dumps(result, indent=4) + "\n"))

@click.group()
def evaluation():
    pass

if __name__ == '__main__':
    evaluation.add_command(test_wise)
    evaluation.add_command(class_wise)
    evaluation.add_command(project_wise)
    evaluation()