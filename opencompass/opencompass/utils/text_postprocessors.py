import re
from typing import Callable, Optional, Union

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('general')
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('general_cn')
def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    import jieba

    cleaned_text = ' '.join(jieba.cut(text))
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('first-capital')
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


@TEXT_POSTPROCESSORS.register_module('last-capital')
def last_capital_postprocess(text: str) -> str:
    for t in text[::-1]:
        if t.isupper():
            return t
    return ''


@TEXT_POSTPROCESSORS.register_module('think_pred')
def think_pred_postprocess(
    prediction: str,
    re_pattern: str,
) -> str:
    match = re.search(re_pattern, prediction)
    if match:
        return match.group(1).strip()
    else:
        return prediction


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        rf'答案是?\s*([{options}])',
        rf'答案是?\s*：\s*([{options}])',
        rf'答案是?\s*:\s*([{options}])',
        rf'答案是选项?\s*:\s*([{options}])',
        rf'答案选项应?该?是\s*([{options}])',
        rf'答案选项应?该?为\s*([{options}])',
        rf'答案应该?是\s*([{options}])',
        rf'答案应该?选\s*([{options}])',
        rf'答案选项为?\s*：\s*([{options}])',
        rf'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'答案选项是?\s*:\s*([{options}])',
        rf'答案为\s*([{options}])',
        rf'答案选\s*([{options}])',
        rf'选择?\s*([{options}])',
        rf'故选?\s*([{options}])'
        rf'只有选?项?\s?([{options}])\s?是?对',
        rf'只有选?项?\s?([{options}])\s?是?错',
        rf'只有选?项?\s?([{options}])\s?不?正确',
        rf'只有选?项?\s?([{options}])\s?错误',
        rf'说法不?对选?项?的?是\s?([{options}])',
        rf'说法不?正确选?项?的?是\s?([{options}])',
        rf'说法错误选?项?的?是\s?([{options}])',
        rf'([{options}])\s?是正确的',
        rf'([{options}])\s?是正确答案',
        rf'选项\s?([{options}])\s?正确',
        rf'所以答\s?([{options}])',
        rf'所以\s?([{options}][.。$]?$)',
        rf'所有\s?([{options}][.。$]?$)',
        rf'[\s，：:,]([{options}])[。，,\.]?$',
        rf'[\s，,：:][故即]([{options}])[。\.]?$',
        rf'[\s，,：:]因此([{options}])[。\.]?$',
        rf'[是为。]\s?([{options}])[。\.]?$',
        rf'因此\s?([{options}])[。\.]?$',
        rf'显然\s?([{options}])[。\.]?$',
        rf'答案是\s?(\S+)(?:。|$)',
        rf'答案应该是\s?(\S+)(?:。|$)',
        rf'答案为\s?(\S+)(?:。|$)',
        rf'(?i)ANSWER\s*:\s*([{options}])',
        rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        rf'^选项\s?([{options}])',
        rf'^([{options}])\s?选?项',
        rf'(\s|^)[{options}][\s。，,：:\.$]',
        rf'1\.\s?(.*?)$',
        rf'1\.\s?([{options}])[.。$]?$',
        rf'答案:\s*([{options}])',
        rf'答案：\s*([{options}])',
        rf'故此为\s*([{options}])',
        rf'boxed\{{([{options}])\}}',  # boxed{([A-D])}
    ]
    cushion_patterns = [
        rf'([{options}]):',
        rf'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable
    text = text.replace('\n\nAssistant:', '')
    text = text.strip()
    text = text.replace('**', '')

    if cushion:
        patterns.extend(cushion_patterns)
    if not text:
        return ''
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    for t in text:
        if t.isupper() and t in options:
            return t

    if text and text[0] in options:
        return text[0]

    return ''


@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        return match.group(1)
    return ''


def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        return match[-1]
    return ''


def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None


@TEXT_POSTPROCESSORS.register_module('multiple-select')
def multiple_select_postprocess(text: str) -> str:
    ret = set([t for t in text if t.isupper()])
    return ''.join(sorted(ret))


@TEXT_POSTPROCESSORS.register_module('specific-xml-tag')
def xml_tag_postprocessor(text, tag):
    """Extracts content enclosed within a specified XML-style tag from a
    string.

    Args:
        texts: The input string containing XML-style tags.
        tag: The XML-style tag to extract content from (e.g., "<conclude>").  Must include the angle brackets.

    Returns:
        The content enclosed within the specified tag, or None if the tag is not found.
    """

    # Use a regular expression to find the content within the specified tag.  This handles cases where the tag might appear multiple times.
    matches = re.findall(
        rf'{tag}(.*?)</{tag[1:-1]}>', text,
        re.DOTALL)  # re.DOTALL allows . to match newline characters

    if matches:
        # Only keep the last one
        output = matches[-1].strip(
        )  # Extract the content and remove leading/trailing whitespace
    else:
        output = 'NO ANSWER FOUND'

    return output


def general_eval_wrapper_postprocess(text: str,
                                     postprocess: Optional[Union[
                                         str, Callable]] = None,
                                     **kwargs) -> str:
    """Wrapper for eval text repr. Especially for chatglmpro.

    Args:
        text(str): Text to be postprocessed.
        postprocess(Callable, optional): Original post processing function.
            Defaults to None.
        **kwargs: Other necessary kwargs for post processing function.
    """
    try:
        text = eval(text)
    except Exception:
        # in case empty input or other error, skip eval
        pass

    if postprocess:
        if isinstance(postprocess, str):
            postprocess = TEXT_POSTPROCESSORS.get(postprocess)
        return postprocess(text, **kwargs)
    else:
        return text


def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.search(answer_pattern, response_text)
    extracted_answer = match.group(1) if match else ''
    return extracted_answer


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content')
def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
    """
    # If text contains only end token, split by end token and take the last part
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{re.escape(think_start_token)}(.*?){re.escape(think_end_token)}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    return non_reasoning_content


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content-v2')
def extract_non_reasoning_content_v2(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
    """
    # If text contains only end token, split by end token and take the last part
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip('\n')

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{re.escape(think_start_token)}(.*?){re.escape(think_end_token)}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip('\n')
    return non_reasoning_content


@TEXT_POSTPROCESSORS.register_module('double_newline')
def double_newline_postprocess(text: str) -> str:
    """在双换行符处截断文本，只保留第一部分。

    Args:
        text (str): 输入文本

    Returns:
        str: 处理后的文本

    Examples:
        >>> text = "Hello world\n\nThis is a test"
        >>> double_newline_postprocess(text)
        'Hello world'

        >>> text = "Single\nlinebreak\n\nDouble\nlinebreak"
        >>> double_newline_postprocess(text)
        'Single\nlinebreak'
    """
    # 使用双换行符分割文本，只取第一部分
    parts = text.split('\n\n')
    return parts[0].strip() if parts else text


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content-hunyuan')
def extract_non_reasoning_content_hunyuan(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
    """
    answer_content = text
    if think_end_token in text:
        answer_content = text.split(think_end_token)[-1].strip('\n')

    if '<answer>' in answer_content:
        answer_content = answer_content.split('<answer>')[-1].strip('\n')
    if '</answer>' in answer_content:
        answer_content = answer_content.split('</answer>')[0].strip('\n')

    print(f'answer_content:{answer_content}\n\n')

    return answer_content


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content-gptoss')
def extract_non_reasoning_content_gptoss(
    text: str,
    entries_start_token: str = '##<channel>##',
    entries_end_token: str = '##</channel>##',
) -> str:
    """Extract content between entries start and end tokens from text."""
    final_answer = text
    if entries_start_token in final_answer:
        final_answer = final_answer.split(entries_start_token)[-1]

    if entries_end_token in final_answer:
        final_answer = final_answer.split(entries_end_token)[0]

    if '<|channel|>final<|message|>' in final_answer:
        final_answer = final_answer.split('<|channel|>final<|message|>')[-1]
    elif '<|message|>' in final_answer:
        print(
            f'<|channel|>final<|message|> not in final_answer:\n{final_answer}'
        )
        final_answer = final_answer.split('<|message|>')[-1]
        if '<|end|>' in final_answer:
            final_answer = final_answer.split('<|end|>')[0]
    else:
        print(f'<|message|> not in final_answer:\n{final_answer}')
        final_answer = final_answer

    print(f'final text:\n{final_answer}')

    return final_answer
