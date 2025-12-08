import torch
import torch.nn as nn
import numpy as np


class NormalizerXT(nn.Module):
    """
    A class which returns the normalization parameters for x_t.
    """

    def __init__(
        self, data_mean=None, data_std=None, data_count=None, num_bins=100, stop_update_count=1e6, device="cpu"
    ):
        super().__init__()
        self.data_mean_fixed = data_mean
        self.data_std_fixed = data_std
        self.num_bins = num_bins
        self.stop_update_count = stop_update_count
        if self.data_mean_fixed is None:
            self.register_buffer("data_mean", torch.Tensor([0.0] * num_bins).to(device))
            self.register_buffer("data_std", torch.Tensor([1.0] * num_bins).to(device))
            self.register_buffer("count", torch.Tensor([0.0] * num_bins))
            assert self.data_std_fixed is None
            self.count = torch.Tensor([0.0] * num_bins)
        else:
            self.register_buffer("data_mean", torch.Tensor(self.data_mean_fixed).to(device))
            self.register_buffer("data_std", torch.Tensor(self.data_std_fixed).to(device))
            self.register_buffer("count", torch.Tensor([data_count] * num_bins))

    def set_device(self, device):
        self.data_mean = self.data_mean.to(device)
        self.data_std = self.data_std.to(device)
        self.count = self.count.to(device)

    def update(self, x_t, t):
        # assert self.data_mean_fixed is None, "update() should not be called when data_mean is fixed."

        for batch_idx in range(x_t.shape[0]):
            t_bin = int(np.floor(t[batch_idx].item() * self.num_bins))
            if t_bin == self.num_bins:
                # print("Warning: t_bin is at the last bin. for t:", t[batch_idx].item())
                t_bin = self.num_bins - 1

            self.data_mean[t_bin] = (self.data_mean[t_bin] * self.count[t_bin] + x_t[batch_idx].mean()) / (
                1 + self.count[t_bin]
            )
            self.data_std[t_bin] = (self.data_std[t_bin] * self.count[t_bin] + x_t[batch_idx].std()) / (
                1 + self.count[t_bin]
            )
            self.count[t_bin] += 1

    def get_mean_std(self, t: torch.Tensor):
        t_bins = (t * self.num_bins).type(torch.long)
        t_bins[t_bins == self.num_bins] = self.num_bins - 1
        mean_val = self.data_mean[t_bins]
        std_val = self.data_std[t_bins]
        return mean_val, std_val

    def normalize(self, x_t, t, update=False):
        if update and torch.sum(self.count) < self.stop_update_count:
            self.update(x_t, t)

        param_shape = [len(x_t)] + [1] * (len(x_t.shape) - 1)
        # t_bins = (t * self.num_bins).type(torch.long)
        # t_bins[t_bins == self.num_bins] = self.num_bins - 1
        # mean_val = self.data_mean[t_bins].reshape(param_shape).to(x_t.device)
        # std_val = self.data_std[t_bins].reshape(param_shape).to(x_t.device)
        mean_val, std_val = self.get_mean_std(t)
        mean_val = mean_val.reshape(param_shape).to(x_t.device)
        std_val = std_val.reshape(param_shape).to(x_t.device)
        # print(x_t.shape, mean_val.shape, std_val.shape, mean_val.squeeze(), std_val.squeeze())
        return (x_t - mean_val) / std_val
