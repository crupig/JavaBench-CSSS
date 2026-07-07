import os
import json
from typing import Iterable, Dict


def read_file(file: str):
    return open(file, "r").read()


def read_code(
    file: str,
    lang="java",
):
    code_block = "```" + lang + "\n" + read_file(file) + "\n```\n"
    return code_block


def extract_code(code: str, *, rank=1):
    import re

    # matches = re.findall("```java\r?\n(.*?)```", code, re.DOTALL)
    matches = re.findall(r"```(?:java)?\r?\n(.*?)```", code, re.DOTALL) # "java" is optional
    if rank <= len(matches):
        return matches[rank - 1]
    return ""


def stream_jsonl(filename: str) -> Iterable[Dict]:
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    with open(filename, mode) as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n").encode("utf-8"))
