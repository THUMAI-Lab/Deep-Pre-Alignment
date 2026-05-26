import logging
import os

from mmengine.logging import MMLogger
from rich.console import Console
from rich.syntax import Syntax

_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}


class ColoredFormatter(logging.Formatter):
    """自定义的彩色日志格式化器"""
    # 定义颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'  # 重置颜色

    def format(self, record):
        # 获取原始格式化的消息
        log_message = super().format(record)
        # 为日志级别添加颜色
        level_name = record.levelname
        if level_name in self.COLORS:
            # 只给 [levelname] 部分添加颜色
            colored_level = (f'{self.COLORS[level_name]}[{level_name}]'
                             f'{self.RESET}')
            log_message = log_message.replace(f'[{level_name}]',
                                              colored_level)
        return log_message


def get_logger(log_level='INFO', filter_duplicate_level=None) -> MMLogger:
    """Get the logger for OpenCompass.

    Args:
        log_level (str): The log level. Default: 'INFO'. Choices are 'DEBUG',
            'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    if not MMLogger.check_instance_created('OpenCompass'):
        # 创建 MMLogger 实例
        logger = MMLogger.get_instance('OpenCompass',
                                       logger_name='OpenCompass',
                                       log_level=log_level)

        # 自定义格式，包含完整路径和行号，不显示毫秒
        log_format = '%(asctime)s [%(levelname)s]' \
            '[PID:%(process)d] ' \
            '%(pathname)s:%(lineno)d - %(message)s'

        # 为所有 handlers 设置自定义格式，指定时间格式不包含毫秒
        # 使用彩色格式化器，让ERROR等级别显示为红色
        formatter = ColoredFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    else:
        logger = MMLogger.get_instance('OpenCompass')

    if filter_duplicate_level is None:
        # export OPENCOMPASS_FILTER_DUPLICATE_LEVEL=error
        # export OPENCOMPASS_FILTER_DUPLICATE_LEVEL=error,warning
        filter_duplicate_level = os.getenv(
            'OPENCOMPASS_FILTER_DUPLICATE_LEVEL', None)

    if filter_duplicate_level:
        logger.addFilter(
            FilterDuplicateMessage('OpenCompass', filter_duplicate_level))

    return logger


def get_logger_ultraeval(log_level='INFO', log_path=None):
    """获取标准 Python logger，支持多种格式样式
    Args:
        log_level (str): 日志级别
        log_path (str): 日志文件路径
        format_style (str): 格式样式选择
            - 'standard': 标准格式，平衡的信息量
            - 'simple': 简洁格式，最少信息
            - 'detailed': 详细格式，包含函数名
            - 'debug': 调试格式，包含进程/线程信息
            - 'fullpath': 完整路径格式，显示完整文件路径
            - 'minimal': 极简格式，只有消息内容
    """
    name = 'ue'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname).1s %(filename)22s:%(lineno)-3s %(message)s')

    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            break
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_path:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                break
        else:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


class FilterDuplicateMessage(logging.Filter):
    """Filter the repeated message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name, filter_duplicate_level):
        super().__init__(name)
        self.seen: set = set()

        if isinstance(filter_duplicate_level, str):
            filter_duplicate_level = filter_duplicate_level.split(',')

        self.filter_duplicate_level = []
        for level in filter_duplicate_level:
            _level = level.strip().upper()
            if _level not in _nameToLevel:
                raise ValueError(f'Invalid log level: {_level}')
            self.filter_duplicate_level.append(_nameToLevel[_level])

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter the repeated error message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno not in self.filter_duplicate_level:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


def pretty_print_config(cfg):
    """Pretty print config using the rich library."""
    console = Console()
    config_str = cfg.pretty_text
    syntax = Syntax(config_str,
                    'python',
                    theme='solarized-dark',
                    line_numbers=True)
    console.print(syntax)
