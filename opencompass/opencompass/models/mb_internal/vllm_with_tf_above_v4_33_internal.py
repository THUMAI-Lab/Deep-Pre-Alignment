# flake8: noqa
# yapf: disable
from typing import Dict, Optional

from opencompass.registry import MODELS

from ..base import LMTemplateParser
from ..vllm_with_tf_above_v4_33 import VLLMwithChatTemplate


@MODELS.register_module()
class VLLMwithChatTemplateInternal(VLLMwithChatTemplate):
    """An Internal Model wrapper for HuggingFace models designed for chat."""

    def __init__(self,
                 *args,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info('Overriding meta_template with internal version.')
        self.template_parser = LMTemplateParser(meta_template)
