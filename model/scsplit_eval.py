import torch.nn as nn
import torch


class EvaluationModel(nn.Module):
    def __init__(self, netG, aggregated_mixing_ratio_ch1, num_timesteps=1):
        super(EvaluationModel, self).__init__()
        self.netG = netG
        self.t_ch1 = aggregated_mixing_ratio_ch1
        self.N = num_timesteps

    def normalize_target(self, target):
        # This needed to be added for uq_processor.get_err_and_var. It neds normalized target. But shouldn't it be on the dset side?
        return target

    def forward(self, x):
        output = []
        for i in range(len(x)):
            ch1_pred = self.netG.indi1.inference(x[i : i + 1], num_timesteps=self.N, t_float_start=self.t_ch1)
            ch2_pred = self.netG.indi2.inference(x[i : i + 1], num_timesteps=self.N, t_float_start=1 - self.t_ch1)
            pred = torch.cat([ch1_pred, ch2_pred], dim=1)
            output.append(pred)
        if len(output) == 1:
            return output[0]
        else:
            return torch.cat(output, dim=0)
