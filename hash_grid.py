import numpy as np
import torch
import torch.nn as nn


class HashGrid(nn.Module):

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

        # allocate memory for embeddings
        offsets = []
        offset = 0
        for i in range(num_levels):
            res = int(np.ceil(base_resolution * (per_level_scale ** i)))
            params_in_level = min(self.max_params, res ** input_dim)
            params_in_level = int(np.ceil(params_in_level / 32) * 32)
            offset += params_in_level
            offsets.append(offset)
        offsets = torch.tensor(offsets, dtype=torch.int32)
        self.register_buffer('offsets', offsets)

        self.n_params = offset * level_dim
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        torch.nn.init.xavier_uniform_(self.embeddings)

        # construct the bin_mask
        n_neigs = 1 << input_dim
        neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
        dims = np.arange(input_dim, dtype=np.int64).reshape((1, -1))
        bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool)
        self.register_buffer('bin_mask', bin_mask)

        primes = torch.tensor(
            [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737, 122420729, 163227661, 217636919, 290182597], dtype=torch.int64)
        self.register_buffer('primes', primes)

    def fast_hash(self, ind, hash_size):
        d = ind.shape[-1]
        ind = (ind * self.primes[:d]) & 0xFFFFFFFF
        for i in range(1, d):
            ind[..., 0] ^= ind[..., i]
        return ind[..., 0] % hash_size

    def forward(self, inputs):
        '''
        inputs: torch.Tensor, shape [batch_size, input_dim], in range [0, 1]
        return: torch.Tensor, shape [batch_size, num_levels * level_dim]
        '''

        output = []
        for i in range(self.num_levels):
            res = int(np.ceil(self.base_resolution *
                      (self.per_level_scale ** i)))
            x = inputs * res

            xi = x.long()
            xf = x - xi.float().detach()
            xi = xi.unsqueeze(dim=-2)
            xf = xf.unsqueeze(dim=-2)

            neigs = torch.where(self.bin_mask, xi, xi + 1)
            offset = 0 if i == 0 else self.offsets[i - 1]
            params_in_level = self.offsets[i] - offset
            hash_ids = self.fast_hash(neigs, params_in_level) + offset
            neigs_features = self.embeddings[hash_ids]

            weights = torch.where(self.bin_mask, 1 - xf, xf)
            w = weights.prod(dim=-1, keepdim=True)

            output.append(torch.sum(neigs_features * w, dim=-2))

        return torch.cat(output, dim=-1)


if __name__ == "__main__":

    hash_grid = HashGrid(input_dim=2)
    for name, param in hash_grid.named_parameters():
        print(name, param.shape)

    inputs = torch.rand(3, 2)
    outputs = hash_grid(inputs)