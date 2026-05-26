from .cpm_model import *  # noqa: F401, F403
from .huggingface_above_v4_33_internal import *  # noqa: F401, F403
from .internal_utils import *  # noqa: F401, F403
from .minicpm_think_vllm import *  # noqa: F401, F403
from .vllm_speed_with_tf_above_v4_33 import *  # noqa: F401, F403
from .vllm_with_tf_above_v4_33_internal import *  # noqa: F401, F403
from .vllm_with_tf_above_v4_33_mid_truncated import *  # noqa: F401, F403

try:
    from .cpmcu_with_tf_above_v4_33 import \
        CPMCUwithChatTemplate  # noqa: F401, F403
except Exception as e:
    print(f'cpmcu_with_tf_above_v4_33 not found {e}')
try:
    from .megrez_vllm import MegrezVLLM, MegrezVLLMBase  # noqa: F401, F403
except Exception as e:
    print(f'megrez_vllm not found {e}')
try:
    from .gpt_oss import GPTOssVLLM  # noqa: F401, F403
except Exception as e:
    print(f'gpt_oss not found {e}')
