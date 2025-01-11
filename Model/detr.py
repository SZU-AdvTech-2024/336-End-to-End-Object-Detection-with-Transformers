import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.models.detr.modeling_detr import DetrEncoderLayer, DetrDecoderLayer
import torch.nn as nn
from typing import Optional
class PromptTunedEncoderLayer(DetrEncoderLayer):
    def __init__(self, config):
        super().__init__(config)

        # 初始化提示词嵌入为可学习参数
        prompt_length = 12  # 根据需要设置提示词的长度
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_length, config.d_model))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            object_queries: torch.Tensor = None,
            output_attentions: bool = False,
    ):
        # 扩展提示词嵌入的批次维度
        batch_size = hidden_states.size(0)
        prompt_embeddings = self.prompt_embeddings.expand(batch_size, -1, -1)

        # 连接提示词嵌入和隐藏状态
        hidden_states = torch.cat([prompt_embeddings, hidden_states], dim=1)

        # 更新 object_queries
        prompt_object_queries = torch.zeros_like(self.prompt_embeddings)
        prompt_object_queries = prompt_object_queries.expand(batch_size, -1, -1)
        if object_queries is not None:
            object_queries = torch.cat([prompt_object_queries, object_queries], dim=1)  # [batch_size, 960, d_model]
        else:
            object_queries = prompt_object_queries  # [batch_size, 10, d_model]

        # 更新注意力掩码
        if attention_mask is not None:
            prompt_len = self.prompt_embeddings.size(1)

            # 创建一个新的注意力掩码，大小为 [batch_size, 1, seq_len + prompt_len, seq_len + prompt_len]
            new_seq_len = hidden_states.size(1)
            new_attention_mask = torch.zeros(
                (batch_size, 1, new_seq_len, new_seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )

            # 将原始的注意力掩码复制到新的位置
            new_attention_mask[:, :, prompt_len:, prompt_len:] = attention_mask

            attention_mask = new_attention_mask

        # 调用父类的 forward 方法
        extended_output = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )
        # 移除提示词嵌入部分
        if isinstance(extended_output, tuple):
            hidden_state = extended_output[0][:, prompt_len:]
            other_outputs = extended_output[1:]
            output = (hidden_state,) + other_outputs
        else:
            output = extended_output[:, prompt_len:]

        return output


class PromptTunedDecoderLayer(DetrDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        # 初始化提示词嵌入为可学习参数
        prompt_length = 12
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_length, config.d_model))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            object_queries: Optional[torch.Tensor] = None,
            query_position_embeddings: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ):
        # 扩展提示词嵌入的批次维度
        batch_size = hidden_states.size(0)
        prompt_embeddings = self.prompt_embeddings.expand(batch_size, -1, -1)

        # 连接提示词嵌入和隐藏状态
        hidden_states = torch.cat([prompt_embeddings, hidden_states], dim=1)

        # 更新 object_queries
        prompt_object_queries = torch.zeros_like(self.prompt_embeddings)
        prompt_object_queries = prompt_object_queries.expand(batch_size, -1, -1)
        if object_queries is not None:
            object_queries = torch.cat([prompt_object_queries, object_queries], dim=1)  # [batch_size, 960, d_model]
        else:
            object_queries = prompt_object_queries  # [batch_size, 10, d_model]

        # 更新注意力掩码
        if attention_mask is not None:
            prompt_len = self.prompt_embeddings.size(1)
            # 创建一个新的注意力掩码，大小为 [batch_size, 1, seq_len + prompt_len, seq_len + prompt_len]
            new_seq_len = hidden_states.size(1)
            new_attention_mask = torch.zeros(
                (batch_size, 1, new_seq_len, new_seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )

            # 将原始的注意力掩码复制到新的位置
            new_attention_mask[:, :, prompt_len:, prompt_len:] = attention_mask

            attention_mask = new_attention_mask
        # 更新 encoder_attention_mask
        if encoder_attention_mask is not None:
            prompt_len = self.prompt_embeddings.size(1)
            batch_size = encoder_hidden_states.size(0)
            orig_query_seq_len = encoder_attention_mask.size(-2)
            orig_key_seq_len = encoder_attention_mask.size(-1)
            new_query_seq_len = orig_query_seq_len + prompt_len
            new_key_seq_len = orig_key_seq_len + prompt_len

            # 创建新的注意力掩码，初始化为零（不遮蔽）
            new_attention_mask = torch.zeros(
                (batch_size, 1, new_query_seq_len, new_key_seq_len),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device
            )

            # 将原始的注意力掩码复制到新的位置
            new_attention_mask[:, :, prompt_len:, prompt_len:] = encoder_attention_mask
            encoder_attention_mask = new_attention_mask

        # 处理 query_position_embeddings
        if query_position_embeddings is not None:
            prompt_position_embeddings = torch.zeros_like(self.prompt_embeddings)
            prompt_position_embeddings = prompt_position_embeddings.expand(batch_size, -1, -1)
            query_position_embeddings = torch.cat([prompt_position_embeddings, query_position_embeddings], dim=1)

        # 处理 encoder_hidden_states
        if encoder_hidden_states is not None:
            prompt_encoder_hidden = torch.zeros_like(self.prompt_embeddings)
            prompt_encoder_hidden = prompt_encoder_hidden.expand(batch_size, -1, -1)
            encoder_hidden_states = torch.cat([prompt_encoder_hidden, encoder_hidden_states], dim=1)

        # 调用父类的 forward 方法
        extended_output = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        # 移除提示词嵌入部分
        if isinstance(extended_output, tuple):
            hidden_state = extended_output[0][:, prompt_len:]
            other_outputs = extended_output[1:]
            output = (hidden_state,) + other_outputs
        else:
            output = extended_output[:, prompt_len:]

        return output


def prompt_model(id2label):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", id2label=id2label, ignore_mismatched_sizes=True)
    for i in range(len(model.model.encoder.layers)):
        old_layer = model.model.encoder.layers[i]
        new_layer = PromptTunedEncoderLayer(model.config)
        old_state_dict = old_layer.state_dict()
        new_layer.load_state_dict(old_state_dict, strict=False)
        model.model.encoder.layers[i] = new_layer
    for i in range(len(model.model.decoder.layers)):
        old_layer = model.model.decoder.layers[i]
        new_layer = PromptTunedDecoderLayer(model.config)
        old_state_dict = old_layer.state_dict()
        new_layer.load_state_dict(old_state_dict, strict=False)
        model.model.decoder.layers[i] = new_layer
    return model


def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = "prompt_embeddings" in name
    return model
