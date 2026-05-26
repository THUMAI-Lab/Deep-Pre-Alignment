# Copyright (c) 2024, BigCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import ast
import traceback
from functools import lru_cache
from typing import Dict, Generator, List, Optional, Set, Tuple

from tree_sitter import Node
from tree_sitter_languages import get_parser

CLASS_TYPE = 'class_definition'
FUNCTION_TYPE = 'function_definition'
IMPORT_TYPE = ['import_statement', 'import_from_statement']
IDENTIFIER_TYPE = 'identifier'
ATTRIBUTE_TYPE = 'attribute'
RETURN_TYPE = 'return_statement'
EXPRESSION_TYPE = 'expression_statement'
ASSIGNMENT_TYPE = 'assignment'


@lru_cache(maxsize=1000)
def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    lines = text.split('\n')
    longest_line_pair = (0, 0)
    longest_so_far = 0

    # 限制搜索范围，避免过度计算
    search_lines = lines[-min(len(lines), 10000):]
    start_offset = len(lines) - len(search_lines)

    # 添加计数器避免无限循环
    max_iterations = 500000000  # 限制最大迭代次数
    iteration_count = 0

    for i in range(len(search_lines)):
        if iteration_count >= max_iterations:
            print(f'Reached max iterations ({max_iterations}), breaking early')
            break

        for j in range(i + 1, min(i + 10000, len(search_lines))):  # 限制内层循环范围
            iteration_count += 1

            if iteration_count >= max_iterations:
                break

            current_lines = '\n'.join(search_lines[i:j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in search_lines[i:j + 1]
                                     if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (start_offset + i, start_offset + j)

                    # 早期退出：如果找到足够长的代码块就停止
                    if current_length > 200:  # 如果找到超过50行的有效代码就足够了
                        print(f'Found good code block: {current_length} lines'
                              ', stopping early')
                        break
        else:
            continue
        break  # 如果内层循环break了，外层也break

    if longest_line_pair == (0, 0):
        # 如果没找到任何有效代码，返回原始文本的一部分
        return '\n'.join(lines[:min(len(lines), 100)])

    return '\n'.join(lines[longest_line_pair[0]:longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:

    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode('utf8'))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str,
                            call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode('utf8')


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(code: str,
                                 entrypoint: Optional[str] = None) -> str:
    # 首先进行快速预处理，如果代码太长则截断
    if len(code) > 50000:  # 如果代码超过50KB，截断到前50KB
        print(f'Code too long ({len(code)} chars), truncating to 50KB')
        code = code[:50000]

    code = code_extract(code.strip())

    # 再次检查处理后的代码长度
    if len(code) > 20000:  # 如果处理后仍然很长，再次截断
        print('Processed code still long: '
              f'{len(code)} chars, truncating to 20KB')
        code = code[:20000]

    code_bytes = bytes(code, 'utf8')

    try:
        parser = get_parser('python')
        tree = parser.parse(code_bytes)
    except Exception as e:
        print(f'Parser failed: {e}, returning first 1000 chars')
        return code[:1000]  # 解析失败时返回前1000字符

    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    # 限制处理的子节点数量
    max_children = 1000  # 最多处理1000个子节点
    processed_children = 0

    for child in root_node.children:
        processed_children += 1
        if processed_children > max_children:
            print(f'Too many children nodes ({processed_children}), '
                  'stopping early')
            break

        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if name and not (name in class_names or name in variable_names
                             or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if name and not (name in function_names or name in variable_names
                             or name in class_names):
                definition_nodes.append((name, child))
                function_names.add(name)
        elif (child.type == EXPRESSION_TYPE and child.children
              and child.children[0].type == ASSIGNMENT_TYPE):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if name and not (name in variable_names or name in function_names
                             or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    # 限制依赖分析的复杂度
    if entrypoint and len(definition_nodes) <= 200:  # 只在节点数量合理时进行依赖分析
        try:
            name2deps = get_deps(definition_nodes)
            reachable = get_function_dependency(entrypoint, name2deps)
        except Exception as e:
            print(
                f'Dependency analysis failed: {e}, including all definitions')
            reachable = set(name for name, _ in definition_nodes)
    else:
        # 如果太复杂或没有entrypoint，包含所有定义
        reachable = set(
            name
            for name, _ in definition_nodes) if definition_nodes else set()

    sanitized_output = b''

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and reachable and not (name in reachable):
            continue
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'

    if not sanitized_output:
        # 如果没有提取到任何内容，返回原始代码的前部分
        return code[:min(len(code), 2000)]

    try:
        sanitized_output = sanitized_output[:-1].decode('utf8')
    except UnicodeDecodeError:
        print('Unicode decode error, returning original code snippet')
        return code[:min(len(code), 2000)]

    # ad-hoc approach to remove unnecessary lines, but it works
    lines = sanitized_output.splitlines()
    if entrypoint and len(lines) < 500:  # 只在行数合理时进行后处理
        outer_lines = []
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(' '):
                break
            if not lines[i].startswith(' ') and entrypoint in lines[i]:
                outer_lines.append(i)
        if outer_lines:
            sanitized_output = '\n'.join(lines[:outer_lines[-1]])

    return sanitized_output


def extract_code_generation(model_output: str,
                            entrypoint: Optional[str] = None):

    # Extract code according to the entrypoint
    sanitized_code = extract_target_code_or_empty(model_output,
                                                  entrypoint).strip()
    # Fallback to extract first codeblock if sanitized_code is empty
    sanitized_code = code_extract(
        model_output) if not sanitized_code else sanitized_code

    return sanitized_code
