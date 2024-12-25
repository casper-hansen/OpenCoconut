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
        batch_size = input_ids.shape[0]

        if input_ids.shape[1] > 1:
            input_ids = torch.concat(
                [
                    input_ids,
                    torch.tensor(
                        [[self.coconut_config.bot_id]] * batch_size, device=input_ids.device
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

        all_thought_outputs = []

        if past_key_values is None:
            past_key_values = DynamicCache()

        # NOTE: only generate thoughts in the prefilling phase
        if self.coconut_config.stages - 1 > 0 and input_ids.shape[1] > 1:
            num_thoughts = (
                self.coconut_config.stages - 1 * self.coconut_config.continuous_thoughts
            )
            inputs_embeds = self.get_input_embeddings()(input_ids)

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
                )

                # The inputs for the next thought will be the current hidden state
                inputs_embeds = outputs.hidden_states[-1][:, -1:, :]
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones(
                        (inputs_embeds.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ), dim=1)
                past_key_values = outputs.past_key_values

                if self.debug:
                    all_thought_outputs.append(
                        self.hidden_states_to_token(inputs_embeds, lm_head=True)
                    )

            inputs_embeds = self.get_input_embeddings()(
                torch.tensor(
                    [[self.coconut_config.eot_id]] * batch_size, device=inputs_embeds.device
                )
            )

            new_attention_mask = []
            for b in range(batch_size):
                new_attention_mask.append(torch.cat((
                    attention_mask[b],
                    torch.ones(
                        num_thoughts-1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                )))
            attention_mask = torch.stack(new_attention_mask, dim=0)

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

                # The inputs for the next thought will be the current hidden state
                inputs_embeds = outputs.hidden_states[-1][:, -1:, :]
                thought_mask = torch.cat((
                    thought_mask,
                    torch.ones(
                        (inputs_embeds.shape[0], 1),
                        dtype=thought_mask.dtype,
                        device=thought_mask.device,
                    ),
                ), dim=1)
                past_key_values = outputs.past_key_values

                if self.debug:
                    all_thought_outputs.append(
                        self.hidden_states_to_token(inputs_embeds, lm_head=True)
                    )

            inputs_embeds = self.get_input_embeddings()(language_ids)

            # we fix the mask and labels lengths by inserting between <bot><eot>
            insert_indices = (input_ids == self.coconut_config.eot_id).nonzero(as_tuple=True)[1]

            new_attention_mask = []
            new_labels = []
            for b in range(input_ids.shape[0]):
                insert_idx = insert_indices[b]
                new_attention_mask.append(torch.cat((
                    attention_mask[b, :insert_idx],
                    torch.ones(
                        num_thoughts-1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask[b, insert_idx:],
                )))

                new_labels.append(torch.cat((
                    labels[b, :insert_idx],
                    torch.full(
                        (num_thoughts-1,), -100, dtype=labels.dtype, device=labels.device
                    ),
                    labels[b, insert_idx:],
                )))

            # Stack the attention masks and labels along the batch dimension
            attention_mask = torch.stack(new_attention_mask, dim=0)
            labels = torch.stack(new_labels, dim=0)

            # FIXME: cannot reuse past_key_values from generating thoughts
            past_key_values = DynamicCache()

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
