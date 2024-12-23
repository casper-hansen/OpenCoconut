import os
import torch
import logging
from typing import Optional, List, Union
from transformers import (
    Qwen2ForCausalLM,
    DynamicCache,
    PreTrainedTokenizer,
)
from . import CoconutConfig
from ..dataset import split_sequences

logger = logging.getLogger(__name__)


class CoconutQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None
        self.coconut_config: CoconutConfig = CoconutConfig.from_dict(
            config.coconut_config
        )
        self.current_stage = 0
        self.debug = os.environ.get("DEBUG") == "1"

    @torch.no_grad()
    def hidden_states_to_token(self, logits: torch.Tensor, lm_head=False):
        if lm_head:
            logits = self.lm_head(logits)
        if logits.dim() == 3:
            logits.squeeze(0)
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, 3)

        tokens = []

        for prob, token_id in zip(top_probs.squeeze(), top_indices.squeeze()):
            tokens.append(
                {
                    "token": self.tokenizer.decode(token_id.item()),
                    "prob": prob.item(),
                    "token_id": token_id.item(),
                }
            )

        return tokens

    def _print_thought_and_final_tokens(
        self, logits: torch.Tensor, all_thought_outputs: List[torch.Tensor]
    ):
        final_thoughts = []
        final_token = self.hidden_states_to_token(logits)[0]
        for i, sampled_tokens in enumerate(all_thought_outputs):
            tokens_formatted = []
            for j, token in enumerate(sampled_tokens):
                tokens_formatted.append(
                    f"t_{i},{j}: [{token['token'].strip()}] (p: {token['prob']:.3f})"
                )
            final_thoughts.append((" || ").join(tokens_formatted))
        print("\n".join(final_thoughts))
        print(
            f"t_final: [{final_token['token'].strip()}] (p: {final_token['prob']:.3f})"
        )

    def infer_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids.shape[1] > 1:
            input_ids = torch.concat(
                [
                    input_ids,
                    torch.tensor(
                        [[self.coconut_config.bot_id]], device=input_ids.device
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.concat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0], 1, device=attention_mask.device
                    ),
                ],
                dim=1,
            )
            cache_position = torch.cat(
                [cache_position, cache_position[-1:] + 1], dim=-1
            )

        all_thought_outputs = []

        if past_key_values is None:
            past_key_values = DynamicCache()

        # NOTE: only generate thoughts in the prefilling phase
        if self.coconut_config.stages - 1 > 0 and input_ids.shape[1] > 1:
            num_thoughts = (
                self.coconut_config.stages - 1 * self.coconut_config.continuous_thoughts
            )
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Initialize past_key_values and cache_position for the first thought
            cache_position = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

            for t in range(num_thoughts):
                outputs = self.model.forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=None,  # Not directly needed in the thought loop
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=cache_position,
                )

                past_key_values = outputs.past_key_values
                current_hidden = outputs.hidden_states[-1][:, -1:, :]
                # The inputs for the next thought will be the current hidden state
                inputs_embeds = current_hidden
                attention_mask = torch.ones(
                    (current_hidden.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                # Update cache_position for the next thought
                cache_position = torch.tensor(
                    [cache_position[-1] + 1],
                    dtype=cache_position.dtype,
                    device=cache_position.device,
                )

                if self.debug:
                    all_thought_outputs.append(
                        self.hidden_states_to_token(current_hidden, lm_head=True)
                    )

            inputs_embeds = self.get_input_embeddings()(
                torch.tensor(
                    [[self.coconut_config.eot_id]], device=inputs_embeds.device
                )
            )
            cache_position = torch.arange(
                cache_position[-1].item(),
                cache_position[-1].item() + 1,
                dtype=cache_position.dtype,
                device=cache_position.device,
            )
            attention_mask = torch.ones(
                (input_ids.shape[0], cache_position[-1].item()), device=input_ids.device
            )

            # Forward pass with combined embeddings
            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )

            if self.debug:
                self._print_thought_and_final_tokens(
                    outputs.logits, all_thought_outputs
                )

        else:
            # Standard forward pass
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **loss_kwargs,
            )

        return outputs

    def train_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Split sequences into thought and language parts
        (
            thought_ids,
            language_ids,
            thought_mask,
            language_mask,
            thought_labels,
            language_labels,
        ) = split_sequences(input_ids, attention_mask, labels, self.coconut_config)

        all_thought_outputs = []

        if past_key_values is None:
            past_key_values = DynamicCache()

        if self.current_stage > 0:
            num_thoughts = self.current_stage * self.coconut_config.continuous_thoughts
            inputs_embeds = self.get_input_embeddings()(thought_ids)

            # Initialize past_key_values and cache_position for the first thought
            cache_position = torch.arange(
                0, thought_ids.shape[1], dtype=torch.long, device=thought_ids.device
            )

            for t in range(num_thoughts):
                outputs = self.model.forward(
                    input_ids=None,
                    attention_mask=thought_mask,
                    position_ids=None,  # Not directly needed in the thought loop
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=cache_position,
                )

                past_key_values = outputs.past_key_values
                current_hidden = outputs.hidden_states[-1][:, -1:, :]
                # The inputs for the next thought will be the current hidden state
                inputs_embeds = current_hidden
                thought_mask = torch.cat((
                    thought_mask,
                    torch.ones(
                        (current_hidden.shape[0], 1),
                        dtype=thought_mask.dtype,
                        device=thought_mask.device,
                    ),
                ), dim=1)
                # Update cache_position for the next thought
                cache_position = torch.tensor(
                    [cache_position[-1] + 1],
                    dtype=cache_position.dtype,
                    device=cache_position.device,
                )

                if self.debug:
                    all_thought_outputs.append(
                        self.hidden_states_to_token(current_hidden, lm_head=True)
                    )

            inputs_embeds = self.get_input_embeddings()(language_ids)

            # the thoughts are hidden in latent space and always takes up +1 in the sequence length dimension
            # we fix the mask and labels by inserting between <bot><eot>
            insert_idx = (input_ids == self.coconut_config.eot_id).nonzero(as_tuple=True)[1][0]
            
            attention_mask = torch.cat((
                attention_mask[:, :insert_idx],
                torch.zeros(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask[:, insert_idx:],
            ), dim=1)
            labels = torch.cat((
                labels[:, :insert_idx],
                torch.full(
                    (labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device
                ),
                labels[:, insert_idx:],
            ), dim=1)
            cache_position = torch.arange(
                cache_position[-1].item(),
                attention_mask.shape[1],
                dtype=cache_position.dtype,
                device=cache_position.device,
            )

            # Forward pass with combined embeddings
            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=language_labels,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )

            if self.debug:
                tokens = []
                for i, (id, mask, label) in enumerate(
                    zip(input_ids[0].tolist(), attention_mask[0].tolist(), labels[0].tolist())
                ):
                    tokens.append(f"<{self.tokenizer.decode(id)}> ({mask}, {label})")
                    if i == insert_idx:
                        tokens.append(f"<[LATENT THOUGHT]> ({mask}, {label})")
                print(" ".join(tokens))
                self._print_thought_and_final_tokens(
                    outputs.logits, all_thought_outputs
                )
        else:
            # Standard forward pass
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **loss_kwargs,
            )

        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        if self.training:
            forward_fn = self.train_forward
        else:
            forward_fn = self.infer_forward

        outputs = forward_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs,
        )

        return outputs
