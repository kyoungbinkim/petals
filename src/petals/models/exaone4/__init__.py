from petals.models.exaone4.config import DistributedExaone4Config
from petals.models.exaone4.model import (
    DistributedExaone4ForCausalLM,
    DistributedExaone4Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedExaone4Config,
    model=DistributedExaone4Model,
    model_for_causal_lm=DistributedExaone4ForCausalLM
)
