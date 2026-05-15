# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from Lara.model.modules.vlm import get_vlm_model
from Lara.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class QwenActionContext:
    last_hidden: torch.Tensor
    action_tokens: torch.Tensor
    embodied_action_tokens: torch.Tensor


class QwenActionTokenizer(nn.Module):
    """Qwen-VL wrapper for LARA prompt tokens and hidden-state extraction."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        embodied_action_token = self.config.framework.vj2_model.get(
            "embodied_action_token", "<|embodied_action|>"
        )
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token,
        )

        self.action_tokens = action_tokens
        self.replace_prompt: Optional[str] = None
        self.embodied_replace_prompt = embodied_action_token * (
            self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction
        )

    @property
    def hidden_size(self) -> int:
        return self.qwen_vl_interface.model.config.hidden_size

    def configure_world_tokens(self, num_prediction_steps: int) -> None:
        self.replace_prompt = "".join(
            [
                token * self.config.framework.vj2_model.num_action_tokens_per_timestep
                for token in self.action_tokens[:num_prediction_steps]
            ]
        )

    def expand_tokenizer(
        self,
        tokenizer: AutoTokenizer,
        special_action_token: str = "<|action_{}|>",
        max_action_tokens: int = 32,
        embodied_action_token: str = "<|embodied_action|>",
    ):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(
                        f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}."
                    )
            action_token_ids.append(tokenizer.convert_tokens_to_ids(action_token_i))

        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(
                    f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}."
                )
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def encode(
        self,
        images: List[list],
        instructions: List[str],
        prompt_template: str,
        include_embodied_tokens: bool,
    ) -> QwenActionContext:
        if self.replace_prompt is None:
            raise RuntimeError("configure_world_tokens() must be called before encoding.")

        prompt_replace_dict = {"{actions}": self.replace_prompt}
        if include_embodied_tokens:
            prompt_replace_dict["{e_actions}"] = self.embodied_replace_prompt

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            prompt_replace_dict=prompt_replace_dict,
            prompt_template=prompt_template,
        )

        action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device),
        ).nonzero(as_tuple=True)
        embodied_action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor([self.embodied_action_token_id], device=qwen_inputs["input_ids"].device),
        ).nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            batch_size, _, hidden_size = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(
                batch_size, -1, hidden_size
            )
            embodied_action_tokens = last_hidden[
                embodied_action_indices[0], embodied_action_indices[1], :
            ].view(batch_size, -1, hidden_size)

        return QwenActionContext(
            last_hidden=last_hidden,
            action_tokens=action_tokens,
            embodied_action_tokens=embodied_action_tokens,
        )
