import json
import argparse
from core.logger import load_json
from data.time_predictor_dataset import TimePredictorDataset
from core.experiment_directory_setup import get_workdir
from data.split_dataset import DataLocation
from model.ddpm_modules.time_predictor import TimePredictor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.optim import Adam
from core.wandb_logger import WandbLogger
from tqdm import tqdm
import numpy as np
from core.logger import mkdirs
import os
from split import add_git_info
from model.normalizer import NormalizerXT


def get_datasets(opt, tiled_pred=False):
    patch_size = opt['datasets']['patch_size']
    target_channel_idx = opt['datasets'].get('target_channel_idx', None)
    upper_clip = opt['datasets'].get('upper_clip', None)
    max_qval = opt['datasets']['max_qval']
    channel_weights = opt['datasets'].get('channel_weights', None)
    normalize_channels = opt['datasets'].get('normalize_channels', False)
    data_type = opt['datasets']['train']['name']  
    uncorrelated_channels = opt['datasets']['train']['uncorrelated_channels']
    
    assert data_type in ['cifar10', 'Hagen','COSEM_jrc-hela', 'HT_LIF24'], f'Invalid data type: {data_type}'
    if data_type == 'Hagen':
        train_data_location = DataLocation(channelwise_fpath=(opt['datasets']['train']['datapath']['ch0'],
                                                        opt['datasets']['train']['datapath']['ch1']))
        val_data_location = DataLocation(channelwise_fpath=(opt['datasets']['val']['datapath']['ch0'],
                                                        opt['datasets']['val']['datapath']['ch1']))
    elif data_type in ['cifar10', 'HT_LIF24', 'COSEM_jrc-hela']:
        train_data_location = DataLocation(directory=(opt['datasets']['train']['datapath']))
        val_data_location = DataLocation(directory=(opt['datasets']['val']['datapath']))
    else:
        raise ValueError('Invalid data type')
    
    gaussian_noise_std_factor = opt['datasets']['train'].get('gaussian_noise_std_factor', None)

    train_set = TimePredictorDataset(data_type, train_data_location, patch_size, 
                             target_channel_idx=target_channel_idx, 
                                max_qval=max_qval, upper_clip=upper_clip,
                                uncorrelated_channels=uncorrelated_channels,
                                channel_weights=channel_weights,
                             normalization_dict=None, enable_transforms=True,random_patching=True,
                             gaussian_noise_std_factor=gaussian_noise_std_factor,
                             normalize_channels=normalize_channels)

    if not tiled_pred:
        class_obj = TimePredictorDataset 
    else:
        raise NotImplementedError('Tiled prediction not implemented yet')

    val_set = class_obj(data_type, val_data_location, patch_size, target_channel_idx=target_channel_idx,
                        #    normalization_dict=train_set.get_normalization_dict(),
                           max_qval=max_qval,
                            upper_clip=upper_clip,
                            channel_weights=channel_weights,
                           enable_transforms=False,
                            random_patching=False,
                            normalize_channels=normalize_channels)
    return train_set, val_set

def start_training(opt):
    if opt['enable_wandb']:
        import wandb
        add_git_info(opt)
        wandb_logger = WandbLogger(opt, opt['path']['experiment_root'], opt['experiment_name'])
        # wandb.define_metric('validation/val_step')
        # wandb.define_metric('epoch')
        # wandb.define_metric("validation/*", step_metric="val_step")
        # val_step = 0
    else:
        wandb_logger = None

    train_set, val_set = get_datasets(opt, tiled_pred=False)
    model_opt = opt['model']
    model_kwargs = {}
    model_kwargs['scale_augmentation'] = model_opt.get('scale_augmentation', False)
    
    if model_kwargs['scale_augmentation']:
        model_kwargs['scale_augmentation_delta'] = model_opt['scale_augmentation_delta']
    
    model = TimePredictor(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=opt['datasets']['patch_size'],
        **model_kwargs,
        )
    model = model.cuda()

    dummy_normalizer_flag = opt['datasets'].get('normalize_channels', False) is True
    # instantiate the normalizer
    if dummy_normalizer_flag:
        print('--------Dummy Normalizer Activated--------')
        xt_normalizer = None
    else:
        xt_normalizer = NormalizerXT()

    train_loader = DataLoader(train_set, batch_size=opt['datasets']['train']['batch_size'], shuffle=True, num_workers=opt['datasets']['train']['num_workers'])
    val_loader = DataLoader(val_set, batch_size=opt['datasets']['train']['batch_size'], shuffle=False, num_workers=opt['datasets']['train']['num_workers'])

    optimizer = Adam(model.parameters(), lr=opt['train']['optimizer']['lr'])
    lr_scheduler_patience = opt['train']['lr_scheduler_patience']
    # learning rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            patience=lr_scheduler_patience,
                                                            factor=0.5,
                                                            min_lr=1e-6,
                                                            verbose=True)# create loss function.
    if opt['model']['loss_type'] == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif opt['model']['loss_type'] == 'l2':
        loss_fn = torch.nn.MSELoss()

    # tqdm bar with loss 
    epoch_losses = []
    num_epochs = opt['train']['num_epochs']
    best_val_loss = 1e6
    for epoch in range(num_epochs):
        bar = tqdm(enumerate(train_loader))
        loss_arr = []
        for i, (x, t_float) in bar:
            model.train()
            optimizer.zero_grad()
            x = x.cuda()
            t_float = t_float.cuda()
            if xt_normalizer is not None:
                x = xt_normalizer.normalize(x,t_float, update=True)
            
            t_float_pred = model(x)
            loss = loss_fn(t_float_pred, t_float.type(torch.float32))
            loss.backward()
            loss_arr.append(loss.item())
            bar.set_description(f'Ep:{epoch} loss {np.mean(loss_arr)} val_loss {best_val_loss}')
            optimizer.step()
            if wandb_logger is not None:
                wandb_logger.log_metrics({'train_loss_step':loss.item()})

        epoch_losses.append(np.mean(loss_arr))
        scheduler.step(epoch_losses[-1])
        # validation
        model.eval()
        val_losses = []
        for i, (x, t_float) in enumerate(val_loader):
            x = x.cuda()
            t_float = t_float.cuda()
            if xt_normalizer is not None:
                x = xt_normalizer.normalize(x,t_float, update=True)
            t_float_pred = model(x)
            loss = loss_fn(t_float_pred, t_float.type(torch.float32))
            val_losses.append(loss.item())


        if wandb_logger is not None:
            wandb_logger.log_metrics({'val_loss':np.mean(val_losses)})
        # print(f'Ep:{epoch} Val loss {np.mean(val_losses)}')
        # save best model

        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            model_fpath = os.path.join(opt['path']['experiment_root'],'best_time_predictor.pth')
            torch.save(model.state_dict(), model_fpath)
            print('Saved best model', model_fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/splitting_hagen_time_predictor.json')
    parser.add_argument('--rootdir', type=str, default='/group/jug/ashesh/training/diffsplit')
    parser.add_argument('-enable_wandb', action='store_true')
    args = parser.parse_args()
    opt = load_json(args.config)
    opt['enable_wandb'] = args.enable_wandb
    experiment_root, expname = get_workdir(opt, args.rootdir, use_max_version=False)

    opt['path']['experiment_root'] = experiment_root
    opt['experiment_name'] = expname
    
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiment_root, path)
            mkdirs(opt['path'][key])

    start_training(opt)