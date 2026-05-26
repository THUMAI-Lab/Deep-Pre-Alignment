"""Zeroshot Retriever."""

from typing import List, Optional

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.registry import ICL_RETRIEVERS
from opencompass.utils.logging import get_logger

from ..icl_prompt_template import PromptTemplate


@ICL_RETRIEVERS.register_module()
class ZeroRetrieverForPPLOnly(BaseRetriever):
    """Zeroshot Retriever. The retriever returns empty list for all queries.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to ''.
    """

    def __init__(self, dataset, ice_eos_token: Optional[str] = '') -> None:
        super().__init__(dataset, '', ice_eos_token, 0)

    def retrieve(self, id_list: List[int] = None) -> List[List]:
        if id_list is not None:
            get_logger().warning('id_list is not empty, but will be ignored.')
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        return rtr_idx_list

    def generate_prompt_for_generate_task(
            self,
            idx,
            ice,
            gen_field_replace_token='',
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        """Generate the prompt for one test example in generative evaluation
        with `prompt_template`. If `prompt_template` is not provided, the
        `ice_template` will be used to generate the prompt. The token
        represented by `gen_field_replace_token` will not be replaced by the
        generated text, or it will leaks the answer.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            gen_field_replace_token (`str`): The token of the answer in the
                prompt. Defaults to ''.
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
        """
        if ice_template is None and prompt_template is not None:
            return prompt_template.generate_iter_for_pplonly(
                self.test_ds[idx],
                output_field=self.dataset_reader.output_column,
                output_field_replace_token=gen_field_replace_token,
                ice_field_replace_token=ice)
        else:
            raise NotImplementedError(
                'Leaving prompt as empty is not supported')
