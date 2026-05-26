from .exambench import *  # noqa: F401, F403
from .LCBench import *  # noqa: F401, F403
from .mb_gaokao import *  # noqa: F401, F403
from .mb_long_text import *  # noqa: F401, F403
from .mbgaokao2025 import *  # noqa: F401, F403
from .mmlu_redux import *  # noqa: F401, F403
from .scaling_bench import *  # noqa: F401, F403

try:
    from .arena_hard_v2 import *  # noqa: F401, F403
except Exception as e:
    print(f'Error loading arena_hard_v2: {e}')

from .zebra_logic import *  # noqa: F401, F403

try:
    from .internal_base import *  # noqa: F401, F403
except Exception as e:
    print(f'Error loading internal_base: {e}')
