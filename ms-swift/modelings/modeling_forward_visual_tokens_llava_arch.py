import torch
from PIL import Image
from typing import Optional, Union
import json
import os
from datetime import datetime
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Qwen3ForCausalLM,
    Qwen3Config
)
from transformers import Qwen2PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import is_torchdynamo_compiling, ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
)
from .modeling_llava_baseline import LLaVABaselineModelForConditionalGeneration, LLaVABaselineConfig
# Compatibility fix: KwargsForCausalLM doesn't exist in newer transformers versions
# try:
#     from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import KwargsForCausalLM
# except ImportError:
#     from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import TransformersKwargs as KwargsForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from dataclasses import dataclass

from transformers.utils import auto_docstring
from transformers import Qwen2Config

IMG_START_ID = 151652
IMG_PAD_ID = 151655
IMG_END_ID = 151653

IMG_THINKER_PAD_ID = 151655
IMG_THINKER_START_ID = 151652
IMG_THINKER_END_ID = 151653



class ForwardVisualTokensArchConfig(Qwen2Config):
    model_type = "forward_visual_tokens_llava_arch"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"perceiver_config": LLaVABaselineConfig}
    has_no_defaults_at_init = True

    def __init__(
        self,
        perceiver_config=None,
        use_cache=True,
        thinker_name_or_path="Qwen/Qwen3-4B",
        t_tokenizer_name_or_path="Qwen/Qwen3-4B",
        p_tokenizer_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        **kwargs,
    ):
        self.use_cache = use_cache
        self.thinker_name_or_path = thinker_name_or_path
        self.t_tokenizer_name_or_path = t_tokenizer_name_or_path
        self.p_processor_name_or_path = p_tokenizer_name_or_path

        self.image_token_id = IMG_PAD_ID

        if isinstance(perceiver_config, dict):
            self.perceiver_config = LLaVABaselineConfig(**perceiver_config)
        else:
            self.perceiver_config = perceiver_config

        super().__init__(**kwargs)


class ForwardVisualTokensArchPreTrainedModel(Qwen2PreTrainedModel):
    config_class = ForwardVisualTokensArchConfig


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class ForwardVisualTokensArchOutputWithPast(ModelOutput):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None


class ForwardVisualTokensArchModel(ForwardVisualTokensArchPreTrainedModel, GenerationMixin):
    def __init__(self, config: ForwardVisualTokensArchConfig):
        super().__init__(config)

        assert self.config.perceiver_config is not None
        assert self.config.thinker_name_or_path is not None

        assert self.config.p_processor_name_or_path is not None
        assert self.config.t_tokenizer_name_or_path is not None

        self.perceiver = LLaVABaselineModelForConditionalGeneration(self.config.perceiver_config)
        # self.perceiver.gradient_checkpointing_enable()

        self.p_processor = AutoProcessor.from_pretrained(
            self.config.p_processor_name_or_path
        )
        self.p_processor.tokenizer.padding_side = "left"

        thinker_config = Qwen3Config.from_pretrained(self.config.thinker_name_or_path)
        self.thinker = Qwen3ForCausalLM(thinker_config)
        # self.thinker.gradient_checkpointing_enable()

        self.t_tokenizer = AutoTokenizer.from_pretrained(
            self.config.t_tokenizer_name_or_path, padding_side="left"
        )

        self.linear_align_dim = torch.nn.Sequential(
            torch.nn.Linear(#因为是嫁接的模型，必须直接访问里面的language model的config才是真实维度
                self.perceiver.model.vlm.language_model.config.hidden_size, self.perceiver.model.vlm.language_model.config.hidden_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.perceiver.model.vlm.language_model.config.hidden_size, self.thinker.config.hidden_size
            ),
        )

        self.config: ForwardVisualTokensArchConfig

    def get_visual_message_tokens(self):
        size = self.config.visual_bandwidth
        tokens = [f"<im_msg-{i}>" for i in range(size)]
        return tokens

    def get_visual_message_token_ids(self, model):
        tokens = self.get_visual_message_tokens()
        if model == "p":
            ids = self.p_processor.tokenizer.convert_tokens_to_ids(tokens)
        elif model == "t":
            ids = self.t_tokenizer.convert_tokens_to_ids(tokens)
        else:
            raise NotImplementedError
        return ids

    def get_visual_message(self):
        message = "".join(self.get_visual_message_tokens())
        return message

    def chat(self, images, msgs, *args, **kwargs):
        assert len(images) == len(msgs)
        assert args == ()
        assert "max_new_tokens" not in kwargs

        # p_prompt_template = 'Encode the image into {num_feat} tokens, including information related to the question. Here is the question: {question}'
        p_prompt_template = "{question}"
        questions = []
        p_images = []
        p_texts = []

        for i in range(len(images)):
            image = images[i]
            msg_list = msgs[i]

            # print(f'Image-{i}: {image}')
            # print(f'Msg-{i}: {msg_list}')

            if not (len(msg_list) == 1 and msg_list[0]["role"] == "user"):
                raise ValueError(
                    f"Each message list must contain a single user dictionary. Error at index {i}."
                )

            pil_image = (
                Image.open(image).convert("RGB") if isinstance(image, str) else image
            )
            p_images.append(pil_image)

            question = msg_list[0]["content"]
            questions.append(question)

            p_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": p_prompt_template.format(question=question),
                        },
                        # {'type': 'text', 'text': p_prompt_template.format(num_feat=self.config.visual_bandwidth,
                        #                                                   question=question)}
                    ],
                }
                # {'role': 'assisstant', 'content': [
                #     {'type': 'text', 'text': self.get_visual_message()}
                # ]}
            ]
            # print(f'P-Message-{i}: {p_message}')
            p_texts.append(
                self.p_processor.apply_chat_template(
                    p_message, tokenize=False, add_generation_prompt=False
                )
            )

        # print(f'{p_texts=}')
        perceiver_inputs = self.p_processor(
            text=p_texts,
            images=p_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # print('Token IDs of perceiver inputs',
        #       perceiver_inputs['input_ids'].tolist())
        # print('Tokens of perceiver inputs', [
        #       self.p_processor.tokenizer.convert_ids_to_tokens(ids) for ids in perceiver_inputs['input_ids']])

        # t_prompt_template = '{question} Image: ' + self.get_visual_message()
        t_prompt_template = "<image>{question}"
        t_texts = []
        for i in range(len(questions)):
            prompt = t_prompt_template.format(question=questions[i])

            p_input_ids = perceiver_inputs["input_ids"][i].tolist()
            img_start_idx = p_input_ids.index(IMG_START_ID)
            img_end_idx = p_input_ids.index(IMG_END_ID)

            assert img_start_idx < img_end_idx

            prompt = prompt.replace(
                "<image>",
                "<|vision_start|>"
                + "<|image_pad|>" * (img_end_idx - img_start_idx - 1)
                + "<|vision_end|>",
            )
            message = [
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": "<think>\n\n</think>\n\n"}
            ]
            t_texts.append(
                self.t_tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    # message, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    # ))
                )
                + "<think>\n\n"
            )
            # print(f'\n\n##T-Message-{i}: {t_texts[-1]}')

        model_inputs_t = self.t_tokenizer(
            t_texts, return_tensors="pt", padding=True
        ).to(self.thinker.device)

        model_inputs_t["input_ids_of_perceiver"] = perceiver_inputs["input_ids"]
        model_inputs_t["attention_mask_of_perceiver"] = perceiver_inputs[
            "attention_mask"
        ]
        model_inputs_t["pixel_values"] = perceiver_inputs["pixel_values"]
        model_inputs_t["image_grid_thw"] = perceiver_inputs["image_grid_thw"]

        # print(
        #     f'Thinker generation config: {self.thinker.generation_config.to_dict()}')
        thinker_generation_params = kwargs.get("thinker_generation_params", {})
        thinker_generation_params["max_new_tokens"] = thinker_generation_params.get(
            "max_new_tokens", 32768
        )

        assert model_inputs_t["pixel_values"] is not None

        with torch.inference_mode():
            generated_ids_t = self.generate(
                **model_inputs_t,
                **thinker_generation_params,
                eos_token_id=self.t_tokenizer.eos_token_id,
            )
        # print(f'Thinker output ids: {generated_ids_t}')
        # print(
        #     f'Thinker output toks: {[self.t_tokenizer.convert_ids_to_tokens(ids) for ids in generated_ids_t]}')

        final_responses = []
        for i in range(len(msgs)):
            output_ids = generated_ids_t[i][len(model_inputs_t.input_ids[i]) :].tolist()
            try:
                # 寻找 </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                print(
                    f"len output_ids: {len(output_ids)}, subtract {output_ids[::-1].index(151668)}"
                )
            except ValueError:
                index = 0

            thinking_content = self.t_tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")
            # print(f"\n\n##Thinking content-{i}: {thinking_content}")

            print(f"content ids: {output_ids[index:]}")

            content = self.t_tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            final_responses.append(content)
            # print(f"\n\n##Answer content-{i}: {content}")

        # return [x[0] for x in self.generate([image], [msgs], *args, **kwargs)]
        return final_responses

    # NOTE: All inputs should be considered as inputs to thinker
    #       The thinker consumes multimodal data by calling perceiver
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        input_ids_of_perceiver=None,
        attention_mask_of_perceiver=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        assert pixel_values is not None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            input_ids_of_perceiver=input_ids_of_perceiver,
            attention_mask_of_perceiver=attention_mask_of_perceiver,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )
        # print(f'\n@@@@ prepare inputs for generation', f'##${model_inputs["pixel_values"].shape}$##', flush=True)

        # # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        # model_inputs["position_ids"] = None

        assert model_inputs["pixel_values"] is not None
        if cache_position[0] != 0:
            # print(f'Cache hit, skip pixel values encoding', flush=True)
            model_inputs["pixel_values"] = None
            # model_inputs["pixel_values_videos"] = None

        return model_inputs

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_of_perceiver: torch.LongTensor = None,
        attention_mask_of_perceiver: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:

        t_input_ids = input_ids
        del input_ids
                
        if inputs_embeds is None:
            inputs_embeds = self.thinker.get_input_embeddings()(t_input_ids)

            if pixel_values is not None:
                p_msg_st_id = IMG_START_ID
                p_msg_ed_id = IMG_END_ID
                p_msg_st_list = []
                p_msg_ed_list = []
                
                # Iterate over batch: each element may contain multiple images (in packing mode)
                for batch_idx, perceiver_sample_input_ids in enumerate(input_ids_of_perceiver):
                    # Find ALL image start/end tokens in this (potentially packed) sequence
                    st_indices = (perceiver_sample_input_ids == p_msg_st_id).nonzero(
                        as_tuple=True
                    )[0]
                    ed_indices = (perceiver_sample_input_ids == p_msg_ed_id).nonzero(
                        as_tuple=True
                    )[0]
                    samples = (perceiver_sample_input_ids == 151644).nonzero(
                        as_tuple=True
                    )[0]
                    
                    # In packing mode: multiple images per sequence (len(st_indices) = pack_size)
                    # In non-packing mode: one image per sequence (len(st_indices) = 1)
                    assert len(st_indices) >= 1, f"No start token found in perceiver input {batch_idx}"
                    assert len(ed_indices) >= 1, f"No end token found in perceiver input {batch_idx}"
                    assert len(st_indices) == len(ed_indices), f"Mismatched start/end tokens in batch {batch_idx}"
                    
                    # Collect start/end positions for all images in this batch element
                    for st, ed in zip(st_indices, ed_indices):
                        p_msg_st_list.append(st)
                        p_msg_ed_list.append(ed)

                # Prepare perceiver inputs
                perceiver_kwargs = {
                    'input_ids': input_ids_of_perceiver,
                    'pixel_values': pixel_values,
                    'attention_mask': attention_mask_of_perceiver,
                    'image_grid_thw': image_grid_thw,
                    'output_hidden_states': True,
                }
                
                # TEMPORARY: Disable position_ids for perceiver to debug hang issue
                # Add position_ids if available (for packing support)
                position_ids_of_perceiver = kwargs.get('position_ids_of_perceiver')
                if position_ids_of_perceiver is not None:
                    perceiver_kwargs['position_ids'] = position_ids_of_perceiver
                
                out = self.perceiver(**perceiver_kwargs)

                # only keep last layer hidden states, release other layers
                last_layer_hiddens = out.hidden_states[-1]
                # print(f"Perceiver last_layer_hiddens shape: {last_layer_hiddens.shape}")

                # 释放不需要的中间变量，但保留梯度
                if hasattr(out, "hidden_states"):
                    del out.hidden_states  # 释放其他层的隐藏状态
                if hasattr(out, "attentions"):
                    del out.attentions  # 释放注意力权重

                # Extract visual features from all images
                # p_msg_st_list and p_msg_ed_list contain positions for all images in order
                # We need to track which batch element each position belongs to
                batch_msg = []
                img_idx = 0  # Track which image we're processing
                for batch_idx, perceiver_sample_input_ids in enumerate(input_ids_of_perceiver):
                    # Find how many images are in this batch element
                    st_indices = (perceiver_sample_input_ids == p_msg_st_id).nonzero(as_tuple=True)[0]
                    num_images_in_batch = len(st_indices)
                    
                    # Extract features for each image in this batch element
                    for _ in range(num_images_in_batch):
                        st = p_msg_st_list[img_idx]
                        ed = p_msg_ed_list[img_idx]
                        # Extract from the correct batch element's hidden states
                        msg_feat = last_layer_hiddens[batch_idx, st : ed + 1, :]
                        batch_msg.append(msg_feat)
                        img_idx += 1
                # print(f"Extracted {len(batch_msg)} image features from {input_ids_of_perceiver.shape[0]} batch elements")

                image_features = torch.cat(batch_msg, dim=0)
                image_features = self.linear_align_dim(image_features)

                n_msg_features = image_features.shape[0]
                msg_mask = (
                    (t_input_ids == IMG_THINKER_START_ID)
                    | (t_input_ids == IMG_THINKER_END_ID)
                    | (t_input_ids == IMG_THINKER_PAD_ID)
                )
                n_msg_tokens = msg_mask.sum()

                if not is_torchdynamo_compiling() and n_msg_tokens != n_msg_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_msg_tokens}, features {n_msg_features}"
                    )

                mask_unsqueezed = msg_mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)

                image_mask = mask_expanded.to(inputs_embeds.device)
                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

                del last_layer_hiddens, batch_msg, mask_expanded, mask_unsqueezed

        outputs = self.thinker(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = ForwardVisualTokensArchOutputWithPast(
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logits=outputs.logits,
        )
        
        return output if return_dict else output.to_tuple()


class ForwardVisualTokensArchForCausalLM(ForwardVisualTokensArchPreTrainedModel, GenerationMixin):
    def __init__(self, config: ForwardVisualTokensArchConfig):
        super().__init__(config)
        self.model = ForwardVisualTokensArchModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = self.model.thinker.lm_head
        # del self.model.thinker.lm_head

        self.config.eos_token_id = self.model.thinker.generation_config.eos_token_id
        if self.model.t_tokenizer.pad_token_id is None:
            self.model.t_tokenizer.pad_token = self.model.t_tokenizer.eos_token

        self.config.pad_token_id = self.model.t_tokenizer.pad_token_id
        print(
            f"Config eos_token_id: {self.config.eos_token_id}, pad_token_id: {self.config.pad_token_id}"
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.thinker.set_input_embeddings(value)

    def _register_perceiver_embedding_gradient_hook(self):
        try:
            embedding_layer = self.model.perceiver.get_input_embeddings()
            print(
                f"Successfully located Perceiver's embedding layer: {embedding_layer}"
            )

            trainable_token_ids = self.model.get_visual_message_token_ids("p")
            if not trainable_token_ids:
                print(
                    "WARNING: No trainable token IDs found for Perceiver. Hook will not be effective."
                )
                return

            print(f"Target trainable token IDs for Perceiver: {trainable_token_ids}")

            vocab_size, _ = embedding_layer.weight.shape
            mask = torch.zeros_like(embedding_layer.weight)

            for token_id in trainable_token_ids:
                mask[token_id, :] = 1.0

            def grad_mask_hook(grad):
                return grad.mul_(mask)

            embedding_layer.weight.register_hook(grad_mask_hook)

            print("=" * 70)
            print("SUCCESS: PERCEIVER embedding gradient hook has been registered.")
            print(
                f"Only embeddings for the following Perceiver token IDs will be updated: {trainable_token_ids}"
            )
            print("This message should only appear ONCE at the beginning of training.")
            print("=" * 70)

        except Exception as e:
            print(
                f"ERROR: Failed to register Perceiver embedding gradient hook. Reason: {e}"
            )

    # def get_output_embeddings(self):
    #     return self.model.thinker.get_output_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        # For lora training
        kwargs['return_dict'] = True
        return_dict = kwargs.get("return_dict", True)
        
        # print('------------------------------------------------', flush=True)
        # print(f'input_ids: {input_ids.shape}', flush=True)
        outputs = self.model(
            input_ids=input_ids,
            # index=index,
            # return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        loss = None

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.model.prepare_inputs_for_generation(input_ids, **kwargs)

    def chat(self, images, msgs, *args, **kwargs):
        return self.model.chat(images, msgs, *args, **kwargs)
