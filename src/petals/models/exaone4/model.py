from typing import Optional

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.exaone4 import Exaone4ForCausalLM, Exaone4ForSequenceClassification, Exaone4Model, Exaone4PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.exaone4.config import DistributedExaone4Config

logger = get_logger(__name__)


class DistributedExaone4Model(FromPretrainedMixin, PTuneMixin, Exaone4Model):
    """ExaONE4Model, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedExaone4Config

    def __init__(self, config: DistributedExaone4Config, *, dht: Optional[hivemind.DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # The causal mask will be added on the server-side
        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.layers.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
        )

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        # Remove prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]
        
        # Add last hidden state
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            # past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm


class DistributedExaone4ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, Exaone4ForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedExaone4Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedExaone4Model._keys_to_ignore_on_load_unexpected
    _supports_cache_class = True
    config_class = DistributedExaone4Config

    def __init__(self, config: DistributedExaone4Config):
        Exaone4PreTrainedModel.__init__(self, config)
        self.model = DistributedExaone4Model(config)
        self.pretraining_tp = getattr(config, 'pretraining_tp', 1)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        
    def _get_initial_cache_position(self, input_ids_seq_length, device, model_kwargs):
        """
        ExaONE4용 캐시 위치 초기화 메서드 오버라이드
        """
        # print(f"[DEBUG] _get_initial_cache_position 호출됨")
        # print(f"[DEBUG] input_ids_seq_length: {input_ids_seq_length}")
        # print(f"[DEBUG] device: {device}")
        
        # past_key_values가 있는지 확인
        past_key_values = model_kwargs.get("past_key_values", None)
        print(f"[DEBUG] past_key_values type: {type(past_key_values)}")
        
        if past_key_values is None:
            # print(f"[DEBUG] 캐시가 없음 - 기본 동작")
            # 캐시가 없으면 기본 동작
            cache_position = torch.arange(0, input_ids_seq_length, dtype=torch.long, device=device)
            model_kwargs["cache_position"] = cache_position
            # print(f"[DEBUG] 초기 cache_position: {cache_position}")
            return model_kwargs
        
        # 캐시가 있으면 적절히 처리
        # print(f"[DEBUG] 캐시가 있음 - 구조 분석")
        try:
            if isinstance(past_key_values, RemotePastKeyValues):
                # RemotePastKeyValues 처리
                if hasattr(past_key_values, '_seen_tokens'):
                    past_length = past_key_values._seen_tokens
                    # print(f"[DEBUG] RemotePastKeyValues - _seen_tokens: {past_length}")
                elif hasattr(past_key_values, 'get_seq_length'):
                    past_length = past_key_values.get_seq_length()
                    # print(f"[DEBUG] RemotePastKeyValues - get_seq_length: {past_length}")
                else:
                    past_length = 0
                    # print(f"[DEBUG] RemotePastKeyValues - 기본값 사용: {past_length}")
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                if isinstance(past_key_values[0], (list, tuple)) and len(past_key_values[0]) > 0:
                    # 표준 캐시 구조 (key, value)
                    past_length = past_key_values[0][0].shape[2]
                    # print(f"[DEBUG] 표준 캐시 구조 - past_length: {past_length}")
                else:
                    # 다른 캐시 구조
                    # print(f"[DEBUG] 비표준 캐시 구조")
                    past_length = 0
            else:
                # print(f"[DEBUG] 빈 캐시 또는 인식 불가 구조")
                past_length = 0
        except (IndexError, AttributeError) as e:
            # 캐시 구조를 파악할 수 없으면 0으로 설정
            # print(f"[DEBUG] 캐시 구조 파악 실패: {type(e).__name__}: {e}")
            past_length = 0
        
        cache_position = torch.arange(
            past_length, past_length + input_ids_seq_length, dtype=torch.long, device=device
        )
        model_kwargs["cache_position"] = cache_position
        print(f"[DEBUG] 최종 cache_position: {cache_position}")
        return model_kwargs

    # # def _supports_cache_class(self):
    # #     """캐시 클래스 지원 여부"""
    # #     return True
    
    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     **kwargs
    # ):
    #     """생성을 위한 입력 준비"""
    #     # print(f"[DEBUG] prepare_inputs_for_generation 호출됨")
    #     # print(f"[DEBUG] input_ids shape: {input_ids.shape if input_ids is not None else None}")
    #     # print(f"[DEBUG] past_key_values type: {type(past_key_values)}")
    #     # print(f"[DEBUG] past_key_values is None: {past_key_values is None}")
        
    #     # past_key_values 처리
    #     if past_key_values is not None:
    #         # print(f"[DEBUG] past_key_values 존재함")
    #         # print(f"[DEBUG] past_key_values 구조: {type(past_key_values)}")
            
    #         if isinstance(past_key_values, (Cache, RemotePastKeyValues)):
    #             # print(f"[DEBUG] Cache 또는 RemotePastKeyValues 타입으로 처리")
    #             if hasattr(past_key_values, 'get_seq_length'):
    #                 cache_length = past_key_values.get_seq_length()
    #                 # print(f"[DEBUG] get_seq_length() 결과: {cache_length}")
    #             else:
    #                 cache_length = 0
                
    #             if hasattr(past_key_values, '_seen_tokens'):
    #                 past_length = past_key_values._seen_tokens
    #                 # print(f"[DEBUG] _seen_tokens 사용: {past_length}")
    #             elif hasattr(past_key_values, 'seen_tokens'):
    #                 past_length = past_key_values.seen_tokens
    #                 # print(f"[DEBUG] seen_tokens 사용: {past_length}")
    #             else:
    #                 past_length = cache_length
    #                 # print(f"[DEBUG] cache_length 사용: {past_length}")
                
    #             if hasattr(past_key_values, 'get_max_length'):
    #                 max_cache_length = past_key_values.get_max_length()
    #             else:
    #                 max_cache_length = None
                    
    #             # print(f"[DEBUG] RemotePastKeyValues - cache_length: {cache_length}, past_length: {past_length}")
    #         else:
    #             # print(f"[DEBUG] 표준 캐시 구조로 처리 시도")
    #             # past_key_values 구조를 안전하게 처리
    #             try:
    #                 # print(f"[DEBUG] past_key_values 길이: {len(past_key_values) if hasattr(past_key_values, '__len__') else 'N/A'}")
                    
    #                 if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
    #                     first_layer = past_key_values[0]
    #                     # print(f"[DEBUG] first_layer type: {type(first_layer)}")
    #                     # print(f"[DEBUG] first_layer 길이: {len(first_layer) if hasattr(first_layer, '__len__') else 'N/A'}")
                        
    #                     if isinstance(first_layer, (list, tuple)) and len(first_layer) > 0:
    #                         # 표준 키-값 캐시 구조: past_key_values[layer_idx][0/1][batch, num_heads, seq_len, head_dim]
    #                         key_tensor = first_layer[0]
    #                         # print(f"[DEBUG] key_tensor shape: {key_tensor.shape if hasattr(key_tensor, 'shape') else 'N/A'}")
    #                         past_length = key_tensor.shape[2]
    #                         # print(f"[DEBUG] 표준 구조 - past_length: {past_length}")
    #                     else:
    #                         # 다른 구조이거나 빈 경우
    #                         # print(f"[DEBUG] 비표준 구조 또는 빈 first_layer")
    #                         past_length = 0
    #                 else:
    #                     # print(f"[DEBUG] past_key_values가 빈 컨테이너이거나 리스트/튜플이 아님")
    #                     past_length = 0
    #             except (IndexError, AttributeError, TypeError) as e:
    #                 # 구조를 파악할 수 없는 경우 안전하게 0으로 설정
    #                 print(f"[DEBUG] 예외 발생: {type(e).__name__}: {e}")
    #                 past_length = 0
                
    #             cache_length = past_length
    #             max_cache_length = None
    #             # print(f"[DEBUG] 최종 - past_length: {past_length}, cache_length: {cache_length}")

    #         # 입력 길이가 캐시보다 긴 경우에만 잘라내기
    #         if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
    #             # print(f"[DEBUG] attention_mask 잘라내기 전: {attention_mask.shape}")
    #             attention_mask = attention_mask[:, -input_ids.shape[1] :]
    #             # print(f"[DEBUG] attention_mask 잘라내기 후: {attention_mask.shape}")
    #     else:
    #         # print(f"[DEBUG] past_key_values가 None - past_length = 0")
    #         past_length = 0

    #     # position_ids 처리
    #     if inputs_embeds is not None and past_key_values is None:
    #         # print(f"[DEBUG] inputs_embeds 사용")
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         # print(f"[DEBUG] input_ids 사용")
    #         model_inputs = {"input_ids": input_ids.contiguous()}

    #     if cache_position is None:
    #         cache_position = torch.arange(
    #             past_length, past_length + input_ids.shape[1], device=input_ids.device
    #         )
    #         # print(f"[DEBUG] cache_position 생성: {cache_position}")

    #     position_ids = cache_position.unsqueeze(0)
    #     # print(f"[DEBUG] position_ids shape: {position_ids.shape}")
        
    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "cache_position": cache_position,
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #         }
    #     )
        
    #     # print(f"[DEBUG] model_inputs keys: {list(model_inputs.keys())}")
    #     # print(f"[DEBUG] prepare_inputs_for_generation 완료")
    #     return model_inputs
    
    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     **kwargs
    # ):
    #     # print("[DEBUG] prepare_inputs_for_generation 호출됨")
    #     # print(f"[DEBUG] input_ids shape: {input_ids.shape}")
    #     # print(f"[DEBUG] past_key_values type: {type(past_key_values)}")
    #     # print(f"[DEBUG] past_key_values is None: {past_key_values is None}")
        
    #     # past_key_values 처리
    #     past_length = 0
    #     if past_key_values is not None:
    #         # print("[DEBUG] past_key_values 존재함")
    #         # print(f"[DEBUG] past_key_values 구조: {type(past_key_values)}")
            
    #         # RemotePastKeyValues 또는 Cache 처리
    #         if isinstance(past_key_values, (Cache, RemotePastKeyValues)):
    #             # print("[DEBUG] Cache 또는 RemotePastKeyValues 타입으로 처리")
    #             try:
    #                 # seen_tokens 속성 우선 확인
    #                 if hasattr(past_key_values, '_seen_tokens'):
    #                     past_length = past_key_values._seen_tokens
    #                     # print(f"[DEBUG] _seen_tokens 사용: {past_length}")
    #                 elif hasattr(past_key_values, 'seen_tokens'):
    #                     past_length = past_key_values.seen_tokens
    #                     # print(f"[DEBUG] seen_tokens 사용: {past_length}")
    #                 elif hasattr(past_key_values, 'get_seq_length'):
    #                     past_length = past_key_values.get_seq_length()
    #                     # print(f"[DEBUG] get_seq_length() 사용: {past_length}")
    #                 else:
    #                     past_length = 0
    #                     # print("[DEBUG] 시퀀스 길이를 얻을 수 없음, 0으로 설정")
    #             except Exception as e:
    #                 print(f"[DEBUG] Cache 처리 중 예외: {e}")
    #                 past_length = 0
    #         else:
    #             # 표준 캐시 구조 처리 시도
    #             # print("[DEBUG] 표준 캐시 구조로 처리 시도")
    #             try:
    #                 if hasattr(past_key_values, '__len__') and len(past_key_values) > 0:
    #                     # print(f"[DEBUG] past_key_values 길이: {len(past_key_values)}")
    #                     first_layer_cache = past_key_values[0]
    #                     if first_layer_cache is not None and len(first_layer_cache) > 0:
    #                         # past_key_values[layer][0=key,1=value][batch, heads, seq, dim]
    #                         past_length = first_layer_cache[0].shape[2]
    #                         # print(f"[DEBUG] 표준 캐시에서 시퀀스 길이: {past_length}")
    #                     else:
    #                         past_length = 0
    #                         # print("[DEBUG] 첫 번째 레이어 캐시가 비어있음")
    #                 else:
    #                     past_length = 0
    #                     # print("[DEBUG] past_key_values가 빈 컨테이너이거나 리스트/튜플이 아님")
    #             except (IndexError, AttributeError, TypeError) as e:
    #                 print(f"[DEBUG] 표준 캐시 처리 중 예외: {e}")
    #                 past_length = 0
        
    #     # print(f"[DEBUG] 최종 past_length: {past_length}")
        
    #     # *** 핵심 수정: input_ids 처리 로직 ***
    #     # past_length > 0이고 input_ids가 과거 시퀀스보다 긴 경우에만 자르기
    #     if past_length > 0 and input_ids.shape[1] > past_length:
    #         # 새로운 토큰만 추출 (마지막 토큰들만)
    #         input_ids = input_ids[:, past_length:]
    #         # print(f"[DEBUG] input_ids 잘라냄: past_length={past_length}, 새로운 shape={input_ids.shape}")
    #     # else:
    #         # print(f"[DEBUG] input_ids 그대로 사용: shape={input_ids.shape}")
        
    #     # cache_position 처리
    #     if cache_position is None:
    #         if past_length > 0:
    #             cache_position = torch.arange(
    #                 past_length, past_length + input_ids.shape[1], 
    #                 device=input_ids.device, dtype=torch.long
    #             )
    #             # print(f"[DEBUG] cache_position 생성 (증분): {cache_position}")
    #         else:
    #             cache_position = torch.arange(
    #                 0, input_ids.shape[1], 
    #                 device=input_ids.device, dtype=torch.long
    #             )
    #             # print(f"[DEBUG] cache_position 생성 (초기): {cache_position}")
    #     # else:
    #         # print(f"[DEBUG] cache_position 전달받음: {cache_position}")
        
    #     # position_ids 생성
    #     position_ids = cache_position.unsqueeze(0)
    #     # print(f"[DEBUG] position_ids shape: {position_ids.shape}")
        
    #     # attention_mask 처리 개선
    #     if attention_mask is not None:
    #         # print(f"[DEBUG] attention_mask 입력: {attention_mask.shape}")
    #         # attention_mask도 input_ids와 동일하게 처리
    #         if past_length > 0 and attention_mask.shape[1] > input_ids.shape[1]:
    #             attention_mask = attention_mask[:, -input_ids.shape[1]:]
    #             # print(f"[DEBUG] attention_mask 조정: {attention_mask.shape}")
        
    #     model_inputs = {
    #         "input_ids": input_ids,
    #         "position_ids": position_ids,
    #         "cache_position": cache_position,
    #         "past_key_values": past_key_values,
    #         "use_cache": kwargs.get("use_cache", False),
    #         "attention_mask": attention_mask,
    #     }
        
    #     # print(f"[DEBUG] model_inputs keys: {list(model_inputs.keys())}")
    #     # print("[DEBUG] prepare_inputs_for_generation 완료")
        
    #     return model_inputs

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedExaone4Model:  # For compatibility with RemoteGenerationMixin
        return self.model
