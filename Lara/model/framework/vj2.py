# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor

from Lara.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC


class VJ2WorldModel(nn.Module):
    """V-JEPA2 encoder plus action-conditioned latent predictor."""

    def __init__(self, config, action_embed_dim: int):
        super().__init__()
        self.config = config
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        self.tubelet_size = self.vj_encoder.config.tubelet_size
        self.num_prediction_steps = self.config.framework.vj2_model.num_frames // self.tubelet_size - 1
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames // self.tubelet_size,
            img_size=(self.vj_encoder.config.image_size, self.vj_encoder.config.image_size),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2,
            action_embed_dim=action_embed_dim,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )

    def forward(self, batch_videos, action_tokens: torch.Tensor) -> torch.Tensor:
        videos = np.stack(batch_videos)  # [B, V, T, H, W, 3]
        videos = videos.transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, 3, H, W]
        batch_size, num_views, num_frames, channels, height, width = videos.shape
        videos = videos.reshape(batch_size * num_views, num_frames, channels, height, width)

        input_videos = []
        for i in range(batch_size * num_views):
            input_videos.append(
                self.vj_processor(videos=videos[i], return_tensors="pt")["pixel_values_videos"].to(
                    self.vj_encoder.device
                )
            )
        input_videos = torch.cat(input_videos, dim=0)

        with torch.no_grad():
            video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
            video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=num_views, dim=0), dim=2)

        latent_frames = num_frames // self.tubelet_size
        tokens_per_frame = video_embeddings.shape[1] // latent_frames
        input_states = video_embeddings[:, : tokens_per_frame * (latent_frames - 1), :]
        gt_states = video_embeddings[:, tokens_per_frame:, :]

        predicted_states = self.vj_predictor(input_states, action_tokens)
        return F.l1_loss(predicted_states, gt_states, reduction="mean")
