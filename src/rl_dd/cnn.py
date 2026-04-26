from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: Iterable[int],
        grid_size: int,
        frame_stack: int,
        object_channels: int = 4,
    ) -> None:
        super().__init__()
        channels = list(hidden_channels)
        if not channels:
            raise ValueError("CNN architecture requires at least one hidden width.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive.")

        expected_dim = grid_size * grid_size * object_channels * frame_stack
        if input_dim != expected_dim:
            raise ValueError(
                "input_dim does not match grid_size, frame_stack, and object_channels: "
                f"got {input_dim}, expected {expected_dim}."
            )

        self.grid_size = int(grid_size)
        self.frame_stack = int(frame_stack)
        self.object_channels = int(object_channels)

        layers: List[nn.Module] = []
        in_channels = self.frame_stack * self.object_channels
        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    int(out_channels),
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            in_channels = int(out_channels)
        self.model = nn.Sequential(*layers)
        self.output_dim = in_channels * self.grid_size * self.grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        grid = x.reshape(
            batch_size,
            self.frame_stack,
            self.grid_size,
            self.grid_size,
            self.object_channels,
        )
        grid = grid.permute(0, 1, 4, 2, 3).contiguous()
        grid = grid.reshape(
            batch_size,
            self.frame_stack * self.object_channels,
            self.grid_size,
            self.grid_size,
        )
        features = self.model(grid)
        return features.flatten(start_dim=1)
