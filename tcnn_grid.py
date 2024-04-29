import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np


class TCNNGrid(nn.Module):

    def __init__(self, input_dim=3, num_levels=8, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19):
        '''
        input_dim: int, dimension of input
        num_levels: int, number of levels
        level_dim: int, feature dimension of each level
        per_level_scale: int, scale factor between levels
        base_resolution: int, resolution of the base level
        log2_hashmap_size: int, log2 of the size of the hash map
        '''

        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size

        self.output_dim = num_levels * level_dim
        self.max_params = 2 ** log2_hashmap_size

        # build tcnn grid
        grid_config = {
            "otype": "HashGrid",
            "n_dims_to_encode": input_dim,
            "n_levels": num_levels,
            "n_features_per_level": level_dim,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
            "log2_hashmap_size": log2_hashmap_size,
            "interpolation": "Linear",
        }
        self.grid = tcnn.Encoding(self.input_dim, grid_config)

    def forward(self, inputs):
        '''
        inputs: torch.Tensor, shape [batch_size, input_dim], in range [0, 1]
        return: torch.Tensor, shape [batch_size, num_levels * level_dim]
        '''

        return self.grid(inputs)
