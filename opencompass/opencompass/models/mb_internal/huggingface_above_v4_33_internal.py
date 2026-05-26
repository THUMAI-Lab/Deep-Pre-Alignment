# TODO: Not fully tested, need to rerun and check the results.
# flake8: noqa
# yapf: disable
import sys
from typing import Dict, List, Optional, Union

import numpy as np

from opencompass.models.base import LMTemplateParser
from opencompass.models.huggingface_above_v4_33 import (
    HuggingFaceBaseModel, HuggingFacewithChatTemplate, _convert_base_messages)
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFacewithChatTemplateInternal(HuggingFacewithChatTemplate):
    """An Internal Model wrapper for HuggingFace models designed for chat."""

    def __init__(self,
                 *args,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info('Overriding meta_template with internal version.')
        self.template_parser = LMTemplateParser(meta_template)


@MODELS.register_module()
class HuggingFaceBaseModelInternal(HuggingFaceBaseModel):
    """An Internal HuggingFace model for PPL inference."""

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 stop_words: Optional[str] = [],
                 ppl_type: str = 'token',
                 answer_token_len: int = -1,
                 answer_token_indices: List[int] = [],
                 **other_kwargs):
        # e.g. answer_token_indices = [1420, 1469, 1419, 1479]
        assert ppl_type in ['token', 'byte']
        self.ppl_type = ppl_type
        self.answer_token_len = answer_token_len
        self.answer_token_indices = answer_token_indices

        super().__init__(
            path=path,
            model_kwargs=model_kwargs,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs,
            peft_path=peft_path,
            peft_kwargs=peft_kwargs,
            tokenizer_only=tokenizer_only,
            generation_kwargs=generation_kwargs,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            stop_words=stop_words,
            **other_kwargs)

    def get_ppl_from_template_internal(self,
                                       templates,
                                       mask_length=None,
                                       curr_byte_len=None):
        """Get perplexity given a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            mask_length (List[int]): A list of mask lengths. If provided, the
                perplexity will be calculated only on the unmasked tokens.
        """
        inputs = self.parse_template(templates, mode='ppl')
        return self.get_ppl_internal(inputs, mask_length, curr_byte_len)

    def get_ppl_internal(self,
                         inputs: List[str],
                         mask_length: Optional[List[int]] = None,
                         curr_byte_len: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        if self.ppl_type == 'byte':
            assert curr_byte_len is not None, 'curr_byte_len is required for byte-level PPL calculation.'

        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F
        pad_token_id = self.tokenizer.pad_token_id

        if self.tokenizer.truncation_side != 'right':
            print('Warning: truncation_side is not right, hard set to right.')
            self.tokenizer.truncation_side = 'right'
        if self.tokenizer.padding_side != 'right':
            print('Warning: padding_side is not right, hard set to right.')
            self.tokenizer.padding_side = 'right'

        messages = _convert_base_messages(inputs)
        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )
        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens, use_cache=False)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            import numpy as np
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)
        if self.ppl_type == 'token':
            ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        elif self.ppl_type == 'byte':
            byte_len = np.array(curr_byte_len)
            ce_loss = loss.float().sum(-1).cpu().detach().numpy() / byte_len
        return ce_loss

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F

        if mask_length is not None:
            if self.tokenizer.truncation_side != 'right':
                print('Warning: truncation_side is not right, hard set to right.')
                self.tokenizer.truncation_side = 'right'
            if self.tokenizer.padding_side != 'right':
                print('Warning: padding_side is not right, hard set to right.')
                self.tokenizer.padding_side = 'right'
        elif self.answer_token_len > 0:
            if self.tokenizer.truncation_side != 'left':
                print('Warning: truncation_side is not left, hard set to left.')
                self.tokenizer.truncation_side = 'left'
            if self.tokenizer.padding_side != 'left':
                print('Warning: padding_side is not left, hard set to left.')
                self.tokenizer.padding_side = 'left'

        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)
        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )
        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens, use_cache=False)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()

        if len(self.answer_token_indices) > 0:
            indices = self.answer_token_indices  # 替换为你实际的词索引

            # 提取最后一个 token 的 logits
            last_token_logits = shift_logits[0, -1, :]

            # 提取感兴趣的词的 logits
            selected_logits = last_token_logits[indices]    # torch.Size([4])

            # 计算这些词的 Softmax 概率
            selected_probs = F.softmax(selected_logits, dim=0)
            return selected_probs

        elif self.answer_token_len > 0:
            shift_labels = tokens['input_ids'][:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
                reduction='none').view(batch_size, seq_len - 1)
            lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()
            if self.answer_token_len > 0:
                mask = torch.zeros_like(shift_labels).to(torch.bfloat16)
                mask[:, -self.answer_token_len] = 1
                loss = loss * mask
                lens = self.answer_token_len
            elif mask_length is not None:
                mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
                for i in range(len(mask)):
                    for j in range(mask_length[i] - 1, len(mask[i])):
                        mask[i][j] = 1
                loss = loss * mask
                lens -= np.array(mask_length)
            ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
            return ce_loss
        else:
            raise NotImplementedError(
                'Please provide `answer_token_indices` or `answer_token_len`.'
            )
