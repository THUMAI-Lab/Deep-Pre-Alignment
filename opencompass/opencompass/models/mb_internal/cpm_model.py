from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opencompass.models.base import BaseModel


class CPMModel(BaseModel):
    """CPM model wrapper for OpenCompass.

    Args:
        path (str): The path to the model.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        generation_kwargs (Dict, optional): The generation kwargs for the
            model. Defaults to dict().
        sync_rank (bool): Whether to sync inputs between ranks. Do not use this
            if you are not familiar with this behavior. Check `sync_inputs`
            function for more details. Defaults to False.
        device (str): Device to load the model on. Defaults to "auto".
        torch_dtype (torch.dtype): The torch dtype to load the model with.
            Defaults to torch.bfloat16.
        trust_remote_code (bool): Whether to trust remote code. Defaults to
            True.
    """
    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Optional[Dict] = None,
                 sync_rank: bool = False,
                 device: str = 'auto',
                 torch_dtype: torch.dtype = torch.bfloat16,
                 mode: str = 'none',
                 trust_remote_code: bool = True):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template,
                         generation_kwargs=generation_kwargs or {},
                         sync_rank=sync_rank)

        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.mode = mode
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=trust_remote_code)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model if not tokenizer_only
        if not tokenizer_only:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=trust_remote_code)
        else:
            self.model = None

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.model is None:
            raise RuntimeError(
                'Model is not initialized. Set tokenizer_only=False.')

        results = []
        for input_text in inputs:
            # Create messages format for chat template
            messages = [{'role': 'user', 'content': input_text}]

            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize input
            model_inputs = self.tokenizer([prompt_text],
                                          return_tensors='pt',
                                          add_special_tokens=False,
                                          padding_side='left',
                                          truncation=True).to(
                                              self.model.device)

            if self.mode == 'mid':
                # Reserve space for the tokens to be generated in the future.
                max_prompt_len = self.max_seq_len - max_out_len

                # Retain the first 0.5 * max_prompt_len tokens
                # and the last 0.5 * max_prompt_len tokens, discarding the
                # middle ones, because the prompts' questions are usually at
                # the beginning or the end.
                # To avoid the warning:
                # This is a friendly reminder - the current text generation
                # call will exceed the model's predefined maximum length.
                # Depending on the model, you may observe exceptions,
                # performance degradation, or nothing at all.
                half_max_prompt_len = max_prompt_len // 2
                if half_max_prompt_len > 0 and model_inputs['input_ids'].shape[
                        1] > max_prompt_len:
                    for key in model_inputs.keys():
                        if model_inputs[key].shape[1] > max_prompt_len:
                            field_values = model_inputs[key]
                            model_inputs[key] = torch.cat(
                                (field_values[:, :half_max_prompt_len],
                                 field_values[:, -half_max_prompt_len:]),
                                dim=1)

            # Generate
            generation_kwargs = self.generation_kwargs.copy()
            generation_kwargs.update({
                'max_new_tokens':
                max_out_len,
                'pad_token_id':
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            })

            model_outputs = self.model.generate(**model_inputs,
                                                **generation_kwargs)

            # Extract new tokens
            output_token_ids = [
                model_outputs[i][len(model_inputs['input_ids'][i]):]
                for i in range(len(model_inputs['input_ids']))
            ]

            # Decode output
            response = self.tokenizer.batch_decode(output_token_ids,
                                                   skip_special_tokens=True)[0]
            results.append(response)

        return results

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out.

        Returns:
            List[float]: A list of perplexity scores.
        """
        if self.model is None:
            raise RuntimeError(
                'Model is not initialized. Set tokenizer_only=False.')

        # This is a placeholder implementation
        # For full perplexity calculation, you would need to implement
        # the forward pass and loss calculation
        raise NotImplementedError(
            'Perplexity calculation not implemented for CPMModel yet.')

    def get_ppl_tokenwise(
            self,
            inputs: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get tokenwise perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out.

        Returns:
            List[float]: A list of perplexity scores.
        """
        if self.model is None:
            raise RuntimeError(
                'Model is not initialized. Set tokenizer_only=False.')

        # This is a placeholder implementation
        # For full tokenwise perplexity calculation, you would need to
        # implement the forward pass and loss calculation
        raise NotImplementedError(
            'Tokenwise perplexity calculation not implemented for '
            'CPMModel yet.'
        )

    def encode(self, prompt: str) -> torch.Tensor:
        """Encode prompt to tokens.

        Args:
            prompt (str): Input string.

        Returns:
            torch.Tensor: Encoded tokens.
        """
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tokens to text.

        Args:
            tokens (torch.Tensor): Input tokens.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

    def to(self, device):
        """Move model to device.

        Args:
            device: Target device.
        """
        if self.model is not None:
            self.model.to(device)
