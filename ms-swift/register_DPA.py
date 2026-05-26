import torch
from swift.llm.model.register import (
    ModelMeta,
    ModelGroup,
    Model,
    register_model,
    get_model_tokenizer_multimodal,
)
from swift.llm.model.model_arch import register_model_arch, MultiModelKeys
from swift.utils import get_logger


logger = get_logger()


register_model_arch(
    MultiModelKeys(
        "thinker_forward_MLP_3Bperceiver",
        language_model=["model.thinker", "lm_head", "model.perceiver.lm_head", "model.perceiver.model.vlm.language_model"],
        vision_tower=["model.perceiver.model.vlm.visual"],
        aligner=["model.linear_align_dim"],
    )
)

register_model(
    ModelMeta(
        model_type="forward_visual_tokens_llava_arch",
        model_groups=[
            ModelGroup(
                [
                    Model(
                        model_path="/path/to/your/init_llava_6k_steps"
                    ),
                ]
            )
        ],
        template="duplex_forward_visual_tokens_arch",
        get_function=get_model_tokenizer_multimodal,
        model_arch='thinker_forward_MLP_3Bperceiver',
        architectures=["ForwardVisualTokensArchForCausalLM"],
        is_multimodal=True,
        torch_dtype=torch.bfloat16,
        requires=["qwen_vl_utils"],
    )
)


logger.info("forward_visual_tokens_llava_arch model registered successfully!")


