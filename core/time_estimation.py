from predtiler.dataset import get_tile_manager
from data.tile_stitcher import stitch_predictions
import numpy as np
import torch

class TimeEstimator:
    def __init__(self, agg_mode='mean', agg_lowq=0.5, agg_highq=0.995, agg_tile_size = 64):
        self.set_agg_params(agg_mode, agg_lowq, agg_highq, agg_tile_size)
    
    def set_agg_params(self, agg_mode, agg_lowq=0.5, agg_highq=0.995, agg_tile_size=64):
        self.mode = agg_mode
        self.lowq = agg_lowq
        self.highq = agg_highq
        self.tile_size = agg_tile_size
        assert self.mode in ['mean','median','mode', 'weighted_by_sum', 'weighted_by_product']

    def predict(self, val_set, xt_normalizer, model, mmse_count, return_input=False):
        # val_set.set_fixed_t(mixing_t)
        pred_arr = []
        inp_arr = []
        dloader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=16,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True)
        for data in dloader:
            inp, t_float = data
            assert torch.std(t_float) < 1e-6, f'{t_float}'
            # assert t_float[0] == mixing_t, f'{t_float} != {mixing_t}'
            inp = inp.cuda()
            t_float = t_float.cuda()
            # if xt_normalizer is dummy then nothing will happen with the following operation.
            inp = xt_normalizer.normalize(inp, t_float)
            if return_input:
                inp_arr.append(inp.cpu().numpy())
            tmp_pred_arr = []
            for _ in range(mmse_count):
                with torch.no_grad():
                    pred = model(inp.cuda())
                    tmp_pred_arr.append(pred.cpu().numpy())
            pred_arr.append(np.median(np.stack(tmp_pred_arr),axis=0))
    
        pred_t = np.concatenate(pred_arr)
        if return_input:
            inp_arr = np.concatenate(inp_arr,axis=0)
            return pred_t, inp_arr
        return pred_t


    def aggregate(self, pred_t, val_set, unmixed_ch1, unmixed_ch2):
        if self.mode == 'mean':
            return np.mean(pred_t)
        elif self.mode == 'median':
            return np.median(pred_t)
        elif self.mode == 'mode':
            pred_t_int = (pred_t*100).astype(int)
            bins = np.arange(0,pred_t_int.max()+1,1)
            idx = np.argmax(np.bincount(pred_t_int))
            return bins[idx]/100
        else:
            assert self.mode in ['weighted_by_sum', 'weighted_by_product']
            orig_tiling_manager = val_set.tile_manager
            tile_manager = get_tile_manager(orig_tiling_manager.data_shape, (1, self.tile_size, self.tile_size), orig_tiling_manager.patch_shape)
            try:
                val_set.tile_manager = tile_manager
                H,W = val_set[0][0].shape[-2:]
                pred_spatial = np.tile(pred_t.reshape(-1,1,1,1), (1,1,H,W))
                pred_tiled = stitch_predictions(pred_spatial, val_set.tile_manager)
                
                tar1 = unmixed_ch1*1.0
                tar2 = unmixed_ch2*1.0
                # lowq = 0.5
                tar1 -= np.quantile(tar1,self.lowq)
                tar2 -= np.quantile(tar2,self.lowq)
                tar1[tar1<0] = 0
                tar2[tar2<0] = 0
                # highq = 0.995
                max1 = np.quantile(tar1,self.highq)
                max2 = np.quantile(tar2,self.highq)
                tar1[tar1>max1] = max1
                tar2[tar2>max2] = max2
                if self.mode == 'weighted_by_sum':
                    probab = tar1  + tar2
                else:
                    assert self.mode == 'weighted_by_product'
                    probab = tar1*tar2
                
                probab = probab/np.sum(probab)
                predicted_t = np.sum(probab[...,None]*pred_tiled)
            finally:
                val_set.tile_manager = orig_tiling_manager

            return predicted_t