from petals.models.qwen.config import DistributedQwenConfig
from petals.models.qwen.model import (
    DistributedQwenForCausalLM,
    DistributedQwenModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedQwenConfig,
    model=DistributedQwenModel,
    model_for_causal_lm=DistributedQwenForCausalLM
)
