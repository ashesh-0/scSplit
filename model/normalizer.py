import torch
import torch.nn as nn
import numpy as np

class NormalizerXT(nn.Module):
    """
    A class which returns the normalization parameters for x_t.
    """
    def __init__(self, data_mean=None, data_std=None, data_count=None, num_bins=100, stop_update_count=1e6):
        super().__init__()
        self.data_mean_fixed = data_mean
        self.data_std_fixed = data_std
        self.num_bins = num_bins
        self.stop_update_count = stop_update_count
        if self.data_mean_fixed is None:
            self.register_buffer("data_mean",torch.Tensor([0.0]*num_bins).cuda())
            self.register_buffer("data_std",torch.Tensor([1.0]*num_bins).cuda())
            self.register_buffer("count",torch.Tensor([0.0]*num_bins))
            assert self.data_std_fixed is None
            self.count = torch.Tensor([0.0]*num_bins)
        else:
            self.register_buffer("data_mean",torch.Tensor(self.data_mean_fixed).cuda())
            self.register_buffer("data_std",torch.Tensor(self.data_std_fixed).cuda())
            self.register_buffer("count",torch.Tensor([data_count]*num_bins))

    def update(self, x_t, t):
        # assert self.data_mean_fixed is None, "update() should not be called when data_mean is fixed."

        for batch_idx in range(x_t.shape[0]):
            t_bin = int(np.floor(t[batch_idx].item()*self.num_bins))
            if t_bin == self.num_bins:
                # print("Warning: t_bin is at the last bin. for t:", t[batch_idx].item())
                t_bin = self.num_bins - 1
            
            self.data_mean[t_bin] = (self.data_mean[t_bin]*self.count[t_bin] + x_t[batch_idx].mean())/(1 + self.count[t_bin])
            self.data_std[t_bin] = (self.data_std[t_bin]*self.count[t_bin] + x_t[batch_idx].std())/(1 + self.count[t_bin])
            self.count[t_bin] += 1
    
    def normalize(self, x_t, t, update=False):
        if update and torch.sum(self.count) < self.stop_update_count:
            self.update(x_t, t)
        
        param_shape = [len(x_t)] + [1]*(len(x_t.shape)-1)
        t_bins = (t * self.num_bins).type(torch.long)
        t_bins[t_bins == self.num_bins] = self.num_bins - 1
        mean_val = self.data_mean[t_bins].reshape(param_shape)
        std_val = self.data_std[t_bins].reshape(param_shape)
        return (x_t - mean_val) / std_val