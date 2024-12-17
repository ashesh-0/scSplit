import sys; sys.path.append('../')
import pytest
from model.ddpm_modules.indi import NormalizerXT
import torch
import numpy as np

@pytest.mark.parametrize("num_bins", [10,25])
def test_normalizerXT(num_bins):
    normalizer = NormalizerXT(num_bins=num_bins)
    test_mus = np.arange(-100,-100+num_bins*10, step=10)
    test_std = np.arange(1, 1+num_bins, step=1)
    bins = []
    for _ in range(40000):
        t = np.random.rand()
        random_bin = int(t*num_bins)
        bins.append(random_bin)
        mu = test_mus[random_bin]
        std = test_std[random_bin]
        img = np.random.normal(loc=mu, scale=std, size=(64,64))
        img = img[None]
        t = np.array([t])
        normalizer.update(torch.Tensor(img), torch.Tensor(t))
    
    assert np.abs(normalizer.data_mean - test_mus).max() < 1e-1
    assert np.abs(normalizer.data_std - test_std).max() < 1e-1
    
