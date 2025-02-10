import torch
import torch.nn as nn
import numpy as np

class NormalizerXT(nn.Module):
    """
    A class which returns the normalization parameters for x_t.
    """
    def __init__(self, data_mean=None, data_std=None, data_count=None, num_bins=100, stop_update_count=1e6, q_low=0.005, q_high=0.995):
        super().__init__()
        self.data_mean_fixed = data_mean
        self.data_std_fixed = data_std
        self.data_count_fixed = data_count
        self.num_bins = num_bins
        self.q_low = q_low
        self.q_high = q_high

        self.stop_update_count = stop_update_count
        if self.data_mean_fixed is None:
            self.register_buffer("data_mean",torch.Tensor([0.0]*num_bins).cuda())
            self.register_buffer("data_std",torch.Tensor([1.0]*num_bins).cuda())
            self.register_buffer("count",torch.Tensor([0.0]*num_bins))
            assert self.data_std_fixed is None
            self.count = torch.Tensor([0.0]*num_bins)
        else:
            self.register_buffer("data_mean",self.data_mean_fixed.cuda())
            self.register_buffer("data_std",self.data_std_fixed.cuda())
            self.register_buffer("count",self.data_count_fixed.cuda())

    def update(self, x_t, t):
        qlowval, qhighval = np.quantile(x_t, [self.q_low, self.q_high])
        t_bin = int(np.floor(t*self.num_bins))
        self.data_mean[t_bin] = qlowval
        self.data_std[t_bin] = qhighval - qlowval
        self.count[t_bin] += 1
    
    def normalize(self, x_t, t, update=False):
        # if update and torch.sum(self.count) < self.stop_update_count:
        #     self.update(x_t, t)
        
        param_shape = [len(x_t)] + [1]*(len(x_t.shape)-1)
        t_bins = (t * self.num_bins).type(torch.long)
        mean_val = self.data_mean[t_bins].reshape(param_shape)
        std_val = self.data_std[t_bins].reshape(param_shape)
        return (x_t - mean_val) / std_val