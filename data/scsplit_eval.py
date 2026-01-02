import torch


class EvaluationSyntheticDset:
    def __init__(self, dset, mixing_ratio, normalizer, real_input=False):
        self.dset = dset
        self.mixing_ratio = mixing_ratio
        self.normalizer = normalizer
        self.normalizer.set_device("cpu")
        self._real_input = real_input

        # needed for tiling
        self.tile_manager = self.dset.tile_manager

    def get_mean_std_for_input(self):
        # for poisson noisee handler, we need this function.
        if self._real_input:
            mean, std = self.normalizer.get_mean_std(torch.Tensor([0.955]))
        else:
            mean, std = self.normalizer.get_mean_std(torch.Tensor([self.mixing_ratio]))
        return mean, std

    def __len__(self):
        return len(self.dset)

    def create_input(self, target, mixing_ratio):
        if len(target.shape) == 3:
            inp = target[:1] * (1 - mixing_ratio) + target[1:2] * mixing_ratio
            inp = inp[None]
            norm_inp = self.normalizer.normalize(torch.Tensor(inp), torch.Tensor([mixing_ratio]), update=False)
            norm_inp = norm_inp[0]
        else:
            assert len(target.shape) == 4  # B,C,H,W
            inp = target[:, :1] * (1 - mixing_ratio) + target[:, 1:2] * mixing_ratio
            norm_inp = self.normalizer.normalize(
                torch.Tensor(inp), torch.Tensor([mixing_ratio] * len(inp)), update=False
            )

        return norm_inp

    def __getitem__(self, index):
        data = self.dset[index]
        tar = torch.Tensor(data["target"])
        # normalize tar
        tar = self.normalizer.normalize(tar, torch.Tensor([0, 1.0]), update=False)
        if self._real_input:
            inp = torch.Tensor(data["input"])
            # any t works since it has same stats for each t, for real input
            inp = self.normalizer.normalize(inp, torch.Tensor([0.955]), update=False)
        else:
            inp = self.create_input(data["target"], self.mixing_ratio)
        return inp, tar
