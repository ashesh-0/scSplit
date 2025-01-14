"""
Here, we have two indi models. We also learn a learnable transformation parameter.
"""

from model.ddpm_modules.indi import InDI
import torch.nn as nn
import torch
import numpy as np

class IndiCustomT(InDI):
    def sample_t(self, batch_size, device):
        assert self._t_sampling_mode == 'linear_indi'
        
        # this 0.5 will be potentially changing the game. 
        assert self.num_timesteps % 2 == 0, "num_timesteps should be even since we are dividing it by 2 in the next line."
        maxv = int(self.num_timesteps * 0.5)
        t = torch.randint(1, maxv, (batch_size,),device=device).long()
        alpha = 1/(self._linear_indi_a + 1)
        probab = torch.rand(t.shape, device=device)
        mask_for_max = probab > alpha
        t[mask_for_max] = maxv
        return t / self.num_timesteps

class IndiFullTranslation(InDI):
    def sample_t(self, batch_size, device):
        assert self._t_sampling_mode == 'linear_indi'
        
        # this 0.5 will be potentially changing the game. 
        assert self.num_timesteps % 2 == 0, "num_timesteps should be even since we are dividing it by 2 in the next line."
        maxv = int(self.num_timesteps * 0.5)
        t = torch.randint(1, self.num_timesteps, (batch_size,),device=device).long()
        alpha = 1/(self._linear_indi_a + 1)
        probab = torch.rand(t.shape, device=device)
        mask_for_max = probab > alpha
        t[mask_for_max] = maxv
        return t / self.num_timesteps

def get_complementary_time_predictor(time_predictor):
    class ComplementaryTimePredictor(nn.Module):
        def __init__(self, time_predictor_ch0):
            super().__init__()
            self.time_predictor_ch0 = time_predictor_ch0
        
        def forward(self, x):
            return 1 - self.time_predictor_ch0(x)
    
    return ComplementaryTimePredictor(time_predictor)

class JointIndi(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        out_channel=2,
        lr_reduction=None,
        denoise_fn_ch1=None,
        denoise_fn_ch2=None,
        conditional=True,
        schedule_opt=None,
        val_schedule_opt=None,
        w_input_loss = 0.0,
        e = 0.01,
        allow_full_translation=False,
        time_predictor=None,
        normalize_xt=False,
    ):
        super().__init__()
        assert denoise_fn_ch1 is not None, "denoise_fn_ch1 is not provided."
        assert denoise_fn_ch2 is not None, "denoise_fn_ch2 is not provided."
        assert denoise_fn is None, "denoise_fn is not needed."
        indi_class = IndiCustomT if not allow_full_translation else IndiFullTranslation
        self.time_predictor1 = self.time_predictor2 = None
        if time_predictor is not None:
            self.time_predictor1 = time_predictor
            self.time_predictor2 = get_complementary_time_predictor(time_predictor)

        self.indi1 = indi_class(denoise_fn_ch1, image_size, channels=channels, 
                          loss_type=loss_type, 
                          out_channel = out_channel, 
                          lr_reduction=lr_reduction, 
                          conditional=conditional, 
                          schedule_opt=schedule_opt, 
                          val_schedule_opt=val_schedule_opt, 
                          e=e,
                            time_predictor=self.time_predictor1,
                            normalize_xt=normalize_xt,
                          )

        self.indi2 = indi_class(denoise_fn_ch2, image_size, channels=channels, 
                          loss_type=loss_type, 
                          out_channel = out_channel, 
                          lr_reduction=lr_reduction, 
                          conditional=conditional, 
                          schedule_opt=schedule_opt, 
                          val_schedule_opt=val_schedule_opt, 
                          e=e,
                            time_predictor=self.time_predictor2,
                            normalize_xt=normalize_xt)

        self.val_num_timesteps = self.indi1.val_num_timesteps

        self.alpha_param = nn.Parameter(torch.tensor(0.0))
        self.offset_param = nn.Parameter(torch.tensor(0.0))
        self.scale_param = nn.Parameter(torch.tensor(1.0))
        self.w_input_loss = w_input_loss
        self.w_t_predictor_loss = 1.0
        self.current_log_dict = {}
        with_time_predictor = 'With Time Predictor' if time_predictor is not None else ''
        print(f'[{self.__class__.__name__}]: w_input_loss: {self.w_input_loss}', with_time_predictor)
    
    def get_offset(self):
        return self.offset_param
    
    def get_scale(self):
        return self.scale_param
    
    def get_alpha(self):
        return torch.sigmoid(self.alpha_param)
    
    def get_current_log(self):
        return self.current_log_dict
    
    
    def p_losses(self, x_in, noise=None):
        x_in_ch1 = {'target': x_in['target'][:,0:1], 'input': x_in['target'][:,1:2]}
        x_in_ch2 = {'target': x_in['target'][:,1:2], 'input': x_in['target'][:,0:1]}

        x_recon_ch1, pred_dict1 = self.indi1.get_prediction_during_training(x_in_ch1, noise=noise)    
        x_recon_ch2, pred_dict2 = self.indi2.get_prediction_during_training(x_in_ch2, noise=noise)

        loss_ch1 = self.indi1.loss_func(x_in_ch1['target'], x_recon_ch1)
        loss_ch2 = self.indi2.loss_func(x_in_ch2['target'], x_recon_ch2)
        loss_splitting = (loss_ch1 + loss_ch2) / 2
        loss_input = 0.0
        loss_realinput = 0.0
        loss_t_predictor = 0.0
        if self.time_predictor1 is not None:
            assert self.time_predictor2 is not None, "time_predictor2 is not provided."
            # get the loss for the time predictor
            loss_t1, pred_t1 =self.indi1.time_prediction_loss(pred_dict1['x_clean'], pred_dict1['t_float'])
            loss_t2, pred_t2 = self.indi2.time_prediction_loss(pred_dict2['x_clean'], pred_dict2['t_float'])

            loss_t_predictor = (loss_t1 + loss_t2) / 2

            # real input is one which is not all zeros. 
            real_input = x_in['input']
            valid_mask = ~((real_input ==0).reshape(len(real_input),-1).all(dim=1))
            if valid_mask.sum() > 0:
                x_recon_ch1,_ = self.indi1.get_prediction_during_training({'superimposed_input':real_input[valid_mask]}, use_superimposed_input=True)    
                x_recon_ch2,_ = self.indi2.get_prediction_during_training({'superimposed_input':real_input[valid_mask]}, use_superimposed_input=True)
                
                loss_ch1_realinput = self.indi1.loss_func(x_in_ch1['target'][valid_mask], x_recon_ch1)
                loss_ch2_realinput = self.indi2.loss_func(x_in_ch2['target'][valid_mask], x_recon_ch2)
                loss_realinput = (loss_ch1_realinput + loss_ch2_realinput) / 2

        # self.current_log_dict['loss_input'] = loss_input.item()
        self.current_log_dict['loss_splitting'] = loss_splitting.item()
        self.current_log_dict['alpha'] = self.get_alpha().item()
        self.current_log_dict['offset'] = self.get_offset().item()
        self.current_log_dict['scale'] = self.get_scale().item()
        self.current_log_dict['loss_realinput'] = loss_realinput.item() if isinstance(loss_realinput, torch.Tensor) else 0.0
        self.current_log_dict['loss_t_predictor'] = loss_t_predictor.item() if isinstance(loss_t_predictor, torch.Tensor) else 0.0
        self.current_log_dict['avg_t'] = (pred_t1.mean().item() + pred_t2.mean().item())/2
        self.current_log_dict['std_t'] = (pred_t1.std().item() + pred_t2.std().item())/2
        # print(loss_splitting.item(), loss_realinput.item(), loss_t_predictor.item())
        return loss_splitting + self.w_input_loss*loss_input + loss_realinput + self.w_t_predictor_loss*loss_t_predictor
    

    def _get_t_values(self, t_start, num_timesteps, eps=1e-8):
        t_max = t_start
        step_size = t_max / num_timesteps
        t_min = step_size
        t_values = np.arange(t_max-step_size, t_min-eps, -1*step_size)
        return t_values
    

    @torch.no_grad()
    def inference(self, x_in, continuous=False, num_timesteps=None, t_float_start=0.5, eps=1e-8, infer_time=False):
        t_float_start_ch1 = t_float_start
        t_float_start_ch2 = 1 - t_float_start
        if infer_time:
            t_float_start_ch1 = self.time_predictor1(x_in).cpu().item()
            t_float_start_ch2 = self.time_predictor2(x_in).cpu().item()
        
        ch1 = self.indi1.inference(x_in, continuous=continuous, num_timesteps=num_timesteps, t_float_start=t_float_start_ch1, eps=eps)
        ch2 = self.indi2.inference(x_in, continuous=continuous, num_timesteps=num_timesteps, t_float_start=t_float_start_ch2, eps=eps)
        return torch.cat([ch1, ch2], dim=1)


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    # other functions to make it compatible with the training script
    # they are all taken from model.ddpm_modules.diffusion.GaussianDiffusion
    def set_loss(self, device):
        self.indi1.set_loss(device)
        self.indi2.set_loss(device)
    
    def set_new_noise_schedule(self, schedule_opt, device):
        self.indi1.set_new_noise_schedule(schedule_opt, device)
        self.indi2.set_new_noise_schedule(schedule_opt, device)
    