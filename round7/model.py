from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from round7.data import LANGUAGES


class LanguageConditionedXLSRForCTC(Wav2Vec2PreTrainedModel):
    """One XLS-R encoder and one CTC head, conditioned by known language ID."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        language_codes = tuple(getattr(config, "language_codes", LANGUAGES))
        if language_codes != LANGUAGES:
            raise ValueError(f"config.language_codes must be exactly {list(LANGUAGES)}")
        self.config.language_codes = list(language_codes)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.language_embedding = nn.Embedding(len(language_codes), config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    @classmethod
    def from_xlsr_pretrained(
        cls,
        model_id: str,
        vocab_size: int,
        **from_pretrained_kwargs,
    ) -> "LanguageConditionedXLSRForCTC":
        base = Wav2Vec2Model.from_pretrained(model_id, **from_pretrained_kwargs)
        config = base.config
        config.vocab_size = vocab_size
        config.language_codes = list(LANGUAGES)
        model = cls(config)
        model.wav2vec2.load_state_dict(base.state_dict(), strict=True)
        return model

    def freeze_feature_encoder(self) -> None:
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutput | tuple:
        if language_ids is None:
            raise ValueError("language_ids is required")
        if language_ids.ndim != 1 or language_ids.shape[0] != input_values.shape[0]:
            raise ValueError("language_ids must have shape [batch]")

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        hidden = hidden + self.language_embedding(language_ids).unsqueeze(1)
        logits = self.lm_head(self.dropout(hidden))

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError("label ID exceeds vocabulary size")
            if attention_mask is None:
                input_lengths = torch.full(
                    (input_values.shape[0],),
                    input_values.shape[1],
                    dtype=torch.long,
                    device=input_values.device,
                )
            else:
                input_lengths = attention_mask.sum(-1)
            input_lengths = self._get_feat_extract_output_lengths(input_lengths).to(torch.long)
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            result = (logits, outputs.hidden_states, outputs.attentions)
            return ((loss,) + result) if loss is not None else result
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
