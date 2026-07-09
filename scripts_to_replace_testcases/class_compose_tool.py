# ./app/static_analyzer/class_compose_tool.py
import os
from typing import List, Dict
from tree_sitter import Language, Parser
from tree_sitter_languages import get_parser, get_language
from app.util.io import extract_code

JAVA_LANGUAGE = get_language("java")
QUERY_METHOD = JAVA_LANGUAGE.query("""
(constructor_declaration) @constructor
(method_declaration) @method
""")
QUERY_COMMENT = JAVA_LANGUAGE.query("""
(line_comment) @comment
""")

parser = get_parser("java")

def get_todo_methods(source: str, todo_only=True, error_tolerance=True) -> List[Dict]:
    tree = parser.parse(bytes(source, "utf8"))
    captures = QUERY_METHOD.captures(tree.root_node)
    seqs = {}
    method_declarations = []

    for method_node, _ in captures:
        has_todo = False
        comment_captures = QUERY_COMMENT.captures(method_node)
        for comment_node, _ in comment_captures:
            if 'TODO' in comment_node.text.decode():
                has_todo = True
                break

        name = method_node.child_by_field_name('name').text.decode()
        seq = seqs.get(name, 0)
        seqs[name] = seq + 1

        body_node = method_node.child_by_field_name('body')

        if body_node is not None:
            method_declarations.append({
                'name': name,
                'seq': seq,
                'node': body_node,
                'body_start': body_node.start_byte,
                'body_end': body_node.end_byte,
                'has_todo': has_todo,
            })
    
    if todo_only:
        return list(filter(lambda decl: decl["has_todo"], method_declarations))
    return method_declarations

def retain_todo_method(source: str, method: str, seq: int) -> str:
    todo_methods = get_todo_methods(source)
    for todo_method in reversed(todo_methods):
        if todo_method['name'] != method or todo_method['seq'] != seq:
            source = source[:todo_method['body_start']+1] + source[todo_method['body_end']-1:]
    return source

def retain_todo_method_with_ref(source: str, source_ref: str, todo_method: dict) -> str:
    ref_methods = get_todo_methods(source_ref, todo_only=False)
    todo_methods = get_todo_methods(source)
    for tdm in reversed(todo_methods):
        # if the method is the one to be implemented, leave the TODO instruction
        # otherwise, replace with reference implementation
        if tdm['name'] != todo_method['name'] or tdm['seq'] != todo_method['seq']:
            ref_method = [rm for rm in ref_methods if rm['name'] == tdm['name'] and rm['seq'] == tdm['seq']][0]
            patch = source_ref.splitlines()[ref_method['node'].start_point[0]:ref_method['node'].end_point[0]+1]
            if len(patch) == 1:
                # assume patch is something line ['public int sum(int a, int b) { return a + b; }']
                sign = patch[0].split("{")[0]+"{"
                indent = len(sign) - len(sign.lstrip()) # count indentation spaces before signature
                ret = " " * (indent + 4) + f'{patch[0].split("{")[1].strip(" }")}'
                closing = " " * indent + "}"
                patch = [sign, ret, closing]
            
            # sanitize reference patch from todos
            patch = [line for line in patch if line.strip().startswith("// TODO") == False]
            
            source = source.splitlines()
            source = source[:tdm['node'].start_point[0]+1] + patch[1:] + source[tdm['node'].end_point[0]+1:]
            source = "\n".join(source)

    return source

def replace_method_old(source: str, target: str, method: str, seq: int) -> str:
    source_todo_methods = get_todo_methods(source)
    target_methods = get_todo_methods(target, todo_only=False)

    source_todo_method = next(filter(lambda decl: decl["name"] == method and decl["seq"] == seq, source_todo_methods), None)
    target_method = next(filter(lambda decl: decl["name"] == method and decl["seq"] == seq, target_methods), None)
    patch = ''
    if source_todo_method is not None and target_method is not None and not target_method["node"].has_error:
        patch = target[target_method['body_start']+1 : target_method['body_end']-1]
        source = source[:source_todo_method['body_start']+1] + patch + source[source_todo_method['body_end']-1:]
        signature = source.splitlines()[source_todo_method['node'].start_point[0]]
        patch = signature + patch if patch else ''
    return source, patch

def contains_declaration(source: str, decl_type: str) -> bool:
    tree = parser.parse(source.encode("utf8"))
    root = tree.root_node

    def visit(node, decl_type):
        if node.type == decl_type:
            return True
        for child in node.children:
            if visit(child, decl_type):
                return True
        return False

    return visit(root, decl_type)


def compute_signature(method, seq, source_extracted, source_todo_method):
    if method == "Alive" and seq == 1:
        return (
            "public Alive(@NotNull final Position newPosition, "
            "@NotNull final Position origPosition, "
            "@NotNull final List<Position> collectedGems, "
            "@NotNull final List<Position> collectedExtraLives) {"
        )
    return source_extracted.splitlines()[
        source_todo_method['node'].start_point[0]
    ]

def replace_method(source: str, target: str, method: str, seq: int) -> str:
    source_extracted = extract_code(source)
    source_todo_methods = get_todo_methods(source_extracted)
    if contains_declaration(target, "class_declaration") == False and contains_declaration(target, "method_declaration") == True:
        target = "public class PlacheHolderClass {\n" + target + "\n}"
    
    
    target_methods = get_todo_methods(target, todo_only=False)
    
    # if in the generated patch there are more methods with target_name then take into consideration the
    # original seq, otherwise set seq_target to 0, so that the first implementation is taken
    num_of_target_functions_with_target_name = len([x for x in target_methods if x["name"] == method])
    seq_target = seq if num_of_target_functions_with_target_name > 1 else 0
    
    source_todo_method = next(filter(lambda decl: decl["name"] == method and decl["seq"] == seq, source_todo_methods), None)
    target_method = next(filter(lambda decl: decl["name"] == method and decl["seq"] == seq_target, target_methods), None)
    patch = ''
    # print(target, source_todo_method, target_method, target_method["node"].has_error)
    if source_todo_method is not None and target_method is not None and not target_method["node"].has_error:
        patch = target[target_method['body_start']+1 : target_method['body_end']]
        source_extracted = source_extracted[:source_todo_method['body_start']+1] + patch + source_extracted[source_todo_method['body_end']:]
        signature = compute_signature(method, seq, source_extracted, source_todo_method)
        patch = signature + patch if patch else ''
    
    if source_todo_method is not None and not patch and contains_declaration(target, "method_declaration") == False:
        patch = "\n" + target if target.count("{") < target.count("}") else "\n" + target + "\n}"
        source_extracted = source_extracted[:source_todo_method['body_start']+1] + patch + source_extracted[source_todo_method['body_end']:]
        signature = compute_signature(method, seq, source_extracted, source_todo_method)
        patch = signature + patch
    
    return source_extracted, patch
    