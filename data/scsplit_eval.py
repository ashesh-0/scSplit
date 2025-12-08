import torch


class EvaluationSyntheticDset:
    def __init__(self, dset, mixing_ratio, normalizer):
        self.dset = dset
        self.mixing_ratio = mixing_ratio
        self.normalizer = normalizer
        self.normalizer.set_device("cpu")

        # needed for tiling
        self.tile_manager = self.dset.tile_manager

    def get_mean_std_for_input(self):
        # for poisson noisee handler, we need this function.
        mean, std = self.normalizer.get_mean_std(torch.Tensor([self.mixing_ratio]))
        return mean, std

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        data = self.dset[index]
        inp_indi1 = data["target"][:1] * (1 - self.mixing_ratio) + data["target"][1:2] * self.mixing_ratio
        norm_inp1 = self.normalizer.normalize(
            torch.Tensor(inp_indi1), torch.Tensor([self.mixing_ratio] * len(inp_indi1)), update=False
        )
        tar = torch.Tensor(data["target"])
        # normalize tar
        tar = self.normalizer.normalize(tar, torch.Tensor([0, 1.0]), update=False)
        return norm_inp1, tar
