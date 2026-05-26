import argparse
import contextlib
import io
import json
import multiprocessing
import os
import re
import signal
import tempfile


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


def first_option_postprocess_base(text: str,
                                  options: str,
                                  cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable
    new_patterns = []
    if cushion:
        new_patterns.extend(cushion_patterns)
    new_patterns.extend(patterns)
    for pattern in new_patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([^\n]+)'


def extract_answer(response_text: str):
    # We suggest to return an empty string but not None when extract failed
    match = re.search(ANSWER_PATTERN, response_text)
    return match.group(1) if match else ''


def math_postprocess_v2(text: str) -> str:

    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    for maybe_ans in text.split('.'):
        # if 'final answer' in maybe_ans.lower():
        if re.search('final answer|answer is', maybe_ans.lower()):
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split('.')[0])


def is_equiv(str1, str2):

    def _fix_fracs(string):
        substrs = string.split('\\frac')
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += '\\frac'
                if len(substr) > 0 and substr[0] == '{':
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != '{':
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}{' + b + '}' + post_substr
                        else:
                            new_str += '{' + a + '}{' + b + '}'
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}' + b + post_substr
                        else:
                            new_str += '{' + a + '}' + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split('/')) != 2:
            return string
        a = string.split('/')[0]
        b = string.split('/')[1]
        try:
            a = int(a)
            b = int(b)
            assert string == '{}/{}'.format(a, b)
            new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
            return new_string
        except AssertionError:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing
        # units
        if '\\text{ ' in string:
            splits = string.split('\\text{ ')
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if '\\sqrt' not in string:
            return string
        splits = string.split('\\sqrt')
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != '{':
                a = split[0]
                new_substr = '\\sqrt{' + a + '}' + split[1:]
            else:
                new_substr = '\\sqrt' + split
            new_string += new_substr
        return new_string

    def _fix_sqrt_v2(string):
        _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
        return _string

    def _strip_string(string):
        # linebreaks
        string = string.replace('\n', '')

        # remove inverse spaces
        string = string.replace('\\!', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')

        # remove units (on the right)
        string = _remove_right_units(string)

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        # fix sqrt3 --> sqrt{3}
        string = _fix_sqrt(string)

        # remove spaces
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
        # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = _fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == '0.5':
            string = '\\frac{1}{2}'

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
        # in case the model output is X/Y
        string = _fix_a_slash_b(string)

        return string

    def _strip_string_v2(string):
        string = str(string).strip()
        # linebreaks
        string = string.replace('\n', '')

        # right "."
        string = string.rstrip('.')

        # remove inverse spaces
        string = string.replace('\\!', '')
        string = string.replace('\\ ', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove unit: miles, dollars if after is not none
        _string = re.sub(r'\\text{.*?}$', '', string).strip()
        if _string != '' and _string != string:
            string = _string

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')
        string = string.replace('$', '')

        string = string.replace('\\text', '')
        string = string.replace('x\\in', '')

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605
        string = string.replace('%', '')

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')

        # cdot
        string = string.replace('\\cdot', '')

        # inf
        string = string.replace('infinity', '\\infty')
        if '\\infty' not in string:
            string = string.replace('inf', '\\infty')
        string = string.replace('+\\inity', '\\infty')

        # and
        string = string.replace('and', '')
        string = string.replace('\\mathbf', '')

        # use regex to remove \mbox{...}
        string = re.sub(r'\\mbox{.*?}', '', string)

        # quote
        string.replace("'", '')
        string.replace('"', '')

        # i, j
        if 'j' in string and 'i' not in string:
            string = string.replace('j', 'i')

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
        string = re.sub(r'(\d+)\.0+$', r'\1', string)

        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        string = _fix_sqrt_v2(string)
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
        # Even works with \frac1{72} (but not \frac{72}1).
        # Also does a/b --> \\frac{a}{b}
        string = _fix_fracs(string)

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
        # cases fix in case the model output is X/Y
        string = _fix_a_slash_b(string)

        return string

    if str1 is None and str2 is None:
        print('WARNING: Both None')
        return True
    if str1 is None or str2 is None:
        return False

    strip_string_func = _strip_string_v2

    try:
        ss1 = strip_string_func(str1)
        ss2 = strip_string_func(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    return str1 == str2


def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]


def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


def gsm8k_is_equal(pred, refer):
    try:
        if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
            return True
    except Exception:
        pass
    return False


def bbh_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans


def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0].strip()

    if ans.endswith('.'):
        ans = ans[:-1].strip()

    match = re.search(r'\*\*(.*?)\*\*', ans)
    if match:
        return match.group(1)

    return ans


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


class TimeOutException(Exception):
    pass


def execution(programs, task_id, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
    """

    def _execution(programs, timeout):
        try:
            # Add exec globals to prevent the exec to raise
            # unnecessary NameError for correct answer
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(programs, exec_globals)
            key.append('pass')
        except TimeOutException:
            key.append('timeout')
        except AssertionError:
            key.append('wrong_answer')
        except BaseException as e:
            print(e)
            key.append('failed')

    manager = multiprocessing.Manager()
    key = manager.list()
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        # key might not have value if killed
        return task_id, 'timeout'
    return task_id, key[0]


def _process_answer(text):
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)\s*DONE",
        r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
        r'BEGIN\s*(.*)\s*\[DONE\]',
        r'\[BEGIN\]\s*(.*)\s*DONE',
        r'BEGIN\s*(.*)\s*DONE',
        r'```python\s*(.*)\s*```',
        r'```\s*(.*)\s*```',
        r'```python\s*(.*)\s*$',
        r'```\s*(.*)\s*$',
        r'(.*)\s*```.*',
        r"\[BEGIN\]\s*'(.*)",
        r'\[BEGIN\](.*)',
        r"'(.*)'\s*\[DONE\]",
    ]
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            text = match.group(1)
            break
    text = text.split('```')[0]
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = text.replace('\\_', '_')
    text = text.strip()
    return text


def _process_test(test_case, pred):
    formatted = pred + '\n'
    formatted += test_case
    return formatted


def humaneval_postprocess_v2(text: str) -> str:
    blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
    if len(blocks) >= 1:
        text = blocks[0]
    return text


def eval_mmlu(input_path):
    options = 'ABCD'

    mmlu_subject_dict = {}
    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            subject = data['subject']
            processed_pred = first_option_postprocess(pred, options=options)

            if subject not in mmlu_subject_dict:
                mmlu_subject_dict[subject] = {'correct': 0, 'total': 0}
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1

            mmlu_subject_dict[subject]['correct'] += is_correct
            mmlu_subject_dict[subject]['total'] += 1
            # print("=====")

    accuracy = total_correct / total_num
    mmlu_result_dict = {
        'weight_accuracy': accuracy,
    }

    subject_num = 0
    subject_acc_total = 0
    for subject, value in mmlu_subject_dict.items():
        curr_acc = value['correct'] / value['total']
        mmlu_result_dict[subject] = curr_acc
        subject_acc_total += curr_acc
        subject_num += 1

    mmlu_result_dict['average_accuracy'] = subject_acc_total / subject_num

    return mmlu_result_dict


def eval_mmlu_base(input_path):
    options = 'ABCD'

    mmlu_subject_dict = {}
    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            subject = data['subject']
            processed_pred = first_capital_postprocess(pred)
            # processed_pred = first_option_postprocess_base(pred, options=options)

            if subject not in mmlu_subject_dict:
                mmlu_subject_dict[subject] = {'correct': 0, 'total': 0}
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1

            mmlu_subject_dict[subject]['correct'] += is_correct
            mmlu_subject_dict[subject]['total'] += 1
            # print("=====")

    accuracy = total_correct / total_num
    mmlu_result_dict = {
        'weight_accuracy': accuracy,
    }

    subject_num = 0
    subject_acc_total = 0
    for subject, value in mmlu_subject_dict.items():
        curr_acc = value['correct'] / value['total']
        mmlu_result_dict[subject] = curr_acc
        subject_acc_total += curr_acc
        subject_num += 1

    mmlu_result_dict['average_accuracy'] = subject_acc_total / subject_num

    return mmlu_result_dict


def eval_cmmlu(input_path):
    cmmlu_subject_dict = {}
    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            subject = data['subject']
            processed_pred = first_capital_postprocess(pred)

            if subject not in cmmlu_subject_dict:
                cmmlu_subject_dict[subject] = {'correct': 0, 'total': 0}
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1

            cmmlu_subject_dict[subject]['correct'] += is_correct
            cmmlu_subject_dict[subject]['total'] += 1
            # print("=====")

    accuracy = total_correct / total_num
    cmmlu_result_dict = {
        'weight_accuracy': accuracy,
    }

    subject_num = 0
    subject_acc_total = 0
    for subject, value in cmmlu_subject_dict.items():
        curr_acc = value['correct'] / value['total']
        cmmlu_result_dict[subject] = curr_acc
        subject_acc_total += curr_acc
        subject_num += 1

    cmmlu_result_dict['average_accuracy'] = subject_acc_total / subject_num
    return cmmlu_result_dict


def eval_ceval(input_path):
    ceval_subject_dict = {}
    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            subject = data['subject']
            processed_pred = first_capital_postprocess(pred)

            if subject not in ceval_subject_dict:
                ceval_subject_dict[subject] = {'correct': 0, 'total': 0}
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1

            ceval_subject_dict[subject]['correct'] += is_correct
            ceval_subject_dict[subject]['total'] += 1
            # print("=====")

    accuracy = total_correct / total_num
    ceval_result_dict = {
        'weight_accuracy': accuracy,
    }

    subject_num = 0
    subject_acc_total = 0
    for subject, value in ceval_subject_dict.items():
        curr_acc = value['correct'] / value['total']
        ceval_result_dict[subject] = curr_acc
        subject_acc_total += curr_acc
        subject_num += 1

    ceval_result_dict['average_accuracy'] = subject_acc_total / subject_num
    return ceval_result_dict


def eval_bbh(input_path):

    bbh_multiple_choice_sets = [
        'temporal_sequences',
        'disambiguation_qa',
        'date_understanding',
        'tracking_shuffled_objects_three_objects',
        'penguins_in_a_table',
        'geometric_shapes',
        'snarks',
        'ruin_names',
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_five_objects',
        'logical_deduction_three_objects',
        'hyperbaton',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'movie_recommendation',
        'salient_translation_error_detection',
        'reasoning_about_colored_objects',
    ]
    bbh_free_form_sets = [
        'multistep_arithmetic_two',
        'navigate',
        'dyck_languages',
        'word_sorting',
        'sports_understanding',
        'boolean_expressions',
        'object_counting',
        'formal_fallacies',
        'causal_judgement',
        'web_of_lies',
    ]

    bbh_subject_dict = {}
    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            subject = data['subject']
            if subject in bbh_multiple_choice_sets:
                processed_pred = bbh_mcq_postprocess(pred)
                gold = bbh_mcq_postprocess(gold)
            elif subject in bbh_free_form_sets:
                processed_pred = bbh_freeform_postprocess(pred)
            else:
                print(f'Subject {subject} not found in sets')
            if subject not in bbh_subject_dict:
                bbh_subject_dict[subject] = {'correct': 0, 'total': 0}
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1

            bbh_subject_dict[subject]['correct'] += is_correct
            bbh_subject_dict[subject]['total'] += 1
            # print("=====")

    accuracy = total_correct / total_num
    bbh_result_dict = {
        'weight_accuracy': accuracy,
    }

    subject_num = 0
    subject_acc_total = 0
    for subject, value in bbh_subject_dict.items():
        curr_acc = value['correct'] / value['total']
        bbh_result_dict[subject] = curr_acc
        subject_acc_total += curr_acc
        subject_num += 1

    bbh_result_dict['average_accuracy'] = subject_acc_total / subject_num
    return bbh_result_dict


def eval_arc(input_path):
    total_correct, total_num = 0, 0
    options = 'ABCD'

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            processed_pred = first_option_postprocess(pred, options=options)
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1
            # print("=====")

    accuracy = total_correct / total_num
    arc_result_dict = {
        'average_accuracy': accuracy,
    }

    return arc_result_dict


def eval_arc_base(input_path):
    total_correct, total_num = 0, 0
    options = 'ABCD'

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            processed_pred = first_capital_postprocess(pred)
            # processed_pred = first_option_postprocess_base(pred, options=options)
            is_correct = processed_pred == gold
            total_correct += is_correct
            total_num += 1
            # print("=====")

    accuracy = total_correct / total_num
    arc_result_dict = {
        'average_accuracy': accuracy,
    }

    return arc_result_dict


def eval_math(input_path):

    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            processed_pred = math_postprocess_v2(pred)
            is_correct = is_equiv(processed_pred, gold)
            if is_correct:
                total_correct += 1
            total_num += 1
            # print("=====")

    accuracy = total_correct / total_num
    math_result_dict = {
        'average_accuracy': accuracy,
    }

    return math_result_dict


def eval_gsm8k(input_path):
    total_correct, total_num = 0, 0

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']
            processed_pred = gsm8k_postprocess(pred)
            gold = gsm8k_dataset_postprocess(gold)

            is_correct = gsm8k_is_equal(processed_pred, gold)
            if is_correct:
                total_correct += 1
            total_num += 1
            # print("=====")

    accuracy = total_correct / total_num
    result_dict = {
        'average_accuracy': accuracy,
    }

    return result_dict


def eval_mbpp(input_path):
    total_num = 0

    with open(input_path, 'r') as f:
        curr_id = 0
        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}

        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']

            pred = _process_answer(pred)
            programs = _process_test(gold, pred)
            index = curr_id
            curr_id += 1

            _, ret = execution(programs, index, 10)

            result[ret] += 1
            total_num += 1

        result['average_accuracy'] = result['pass'] / total_num

    return result


def eval_humaneval(input_path):
    from human_eval.data import HUMAN_EVAL, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    k = [1, 10, 100]
    with open(input_path, 'r') as f:
        humaneval_preds = []
        for line in f:
            data = json.loads(line)
            gold = data['gold']
            pred = data['prediction']

            preds = humaneval_postprocess_v2(pred)

            if not isinstance(preds, list):
                preds = [preds]
            for pred in preds:
                humaneval_preds.append({'task_id': gold, 'completion': pred})

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, 'human_eval.json')
            write_jsonl(out_dir, humaneval_preds)
            score = evaluate_functional_correctness(out_dir,
                                                    k,
                                                    n_workers=4,
                                                    timeout=3.0,
                                                    problem_file=HUMAN_EVAL)

            detail_path = os.path.join(tmp_dir,
                                       'human_eval.json_results.jsonl')
            details = {}
            with open(detail_path, 'r') as f:
                for index, line in enumerate(f):
                    line = json.loads(line)
                    line['is_correct'] = line['passed']
                    # line['prompt'] = prompts[index]
                    details[str(index)] = line

        results = {f'humaneval_{k}': score[k] for k in score}
        # results['details'] = details

        results['average_accuracy'] = results['humaneval_pass@1']
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--data', type=str, required=True, help='eval data')
    # parser.add_argument("--input", type=str, default="outputs/processed_inputs/math.jsonl", help="input file")
    # parser.add_argument("--data", type=str, default="math", help="eval data")
    parser.add_argument('--output', type=str, default=None, help='output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output
    eval_data = args.data

    assert os.path.exists(input_path), f'input file {input_path} not exists'
    assert eval_data in ['mmlu', 'cmmlu', 'ceval', 'bbh', 'mmlu_cot', 'arc', 'math', 'gsm8k', 'mbpp', 'humaneval', 'mmlu_base', 'arc_base'], \
        f'eval data {eval_data} not supported'

    if eval_data == 'mmlu_cot':
        # same postprocess function as mmlu
        eval_data = 'mmlu'

    print(f'calculating accuracy for {eval_data} data')
    eval_func_name = f'eval_{eval_data}'
    eval_func = eval(eval_func_name)

    result = eval_func(input_path)
    print(result)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f'result saved to {output_path}')


if __name__ == '__main__':
    main()
