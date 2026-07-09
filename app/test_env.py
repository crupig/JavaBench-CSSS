from io import StringIO
import subprocess
import shutil
import os
import re
import pandas as pd

from typing import List
from app.schema.schemas import CompilerError
from app.static_analyzer.class_compose_tool import get_todo_methods


def to_code_path(root, code_path):
    return os.path.join(root, "src", "main", "java", code_path)

def check_todo(todo_code, com_code):
    todo_methods = get_todo_methods(todo_code)
    com_methods = get_todo_methods(com_code)
    for tm in todo_methods: 
        cm = None
        for _cm in com_methods:
            if _cm['name'] == tm["name"] and _cm['seq'] == tm["seq"]:
                cm = _cm
        if cm == None:
            continue

        tm_body = todo_code[tm["body_start"]:tm["body_end"]]
        cm_body = com_code[cm["body_start"]:cm["body_end"]]
        tm_body = '\n'.join([line for line in tm_body.split('\n') if not line.strip().startswith("//")])
        cm_body = '\n'.join([line for line in cm_body.split('\n') if not line.strip().startswith("//")])
        tm_len = len(tm_body)
        cm_len = len(cm_body)

        if cm_len - tm_len < 10:
            return True
        
    return False

class TestEnv:
    def __init__(self, root: str, todo_src: str, src: str) -> None:
        self.root = root
        self.todo_src = todo_src
        self.src = src
        shutil.copytree(src, root)

    def destroy(self):
        shutil.rmtree(self.root)
    
    def replace(self, target: str, content: str):
        code_path = to_code_path(self.root, target)
        if not os.path.exists(code_path):
            print(f"warning: replace non-existed file {code_path}")

        with open(code_path, "w") as fp:
            fp.write(content)

        return {
            "has_todo": False,
            "can_replace": True
        }

    def recover(self, target: str):
        shutil.copy2(to_code_path(self.src, target), to_code_path(self.root, target))

    def compile(self) -> List[CompilerError]:
        process = subprocess.Popen(
            ["./gradlew", "compileJava", "--info", "-q", "--rerun-tasks"],
            cwd=self.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # env={"LANG": "en_US.UTF-8"},
        )
        _, err = process.communicate()
        errors1 = CompilerError.parse_errors(err_string=err.decode("utf-8"))


        process = subprocess.Popen(
            ["./gradlew", "compileTestJava", "--info", "-q", "--rerun-tasks"],
            cwd=self.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # env={"LANG": "en_US.UTF-8"},
        )
        _, err = process.communicate()
        errors2 = CompilerError.parse_errors(err_string=err.decode("utf-8"))
        return errors1 + errors2

    def run_test(self, target: str | None):
        args = ["./gradlew", "test"]
        if target:
            args += ["--tests", target]
        args += ["--rerun-tasks"]
        process = subprocess.Popen(
            args,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # env={"LANG": "en_US.UTF-8"},
        )
        out, err = process.communicate()
        out, err = out.decode(), err.decode()
        match = re.search(
            r"Results: (\w+) \((\d+) tests, (\d+) successes, (\d+) failures, (\d+) skipped\)",
            out + err,
        )
        return (int(match.group(3)), int(match.group(2))), out, err
    
    def run_dep_metrics(self):
        source_dir = "src/main/java"
        process = subprocess.Popen(
            [
                "java", "-jar", "./lib/java-sellotape.jar", "dep-metric",
                "--generated-project", os.path.join(self.root, source_dir),
                "--solution-project", os.path.join(self.src, source_dir),
                "--todo-project", os.path.join(self.todo_src, source_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"JAVA_TOOL_OPTIONS": "-Duser.language=en", "JAVA_HOME": "/usr/lib/jvm/default-runtime" },
        )
        out, err = process.communicate()
        out = out.decode()
        print(err.decode())
        
        out = pd.read_csv(StringIO(out), header="infer", sep="\s+")
        return out.iloc[0]
