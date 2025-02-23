import numpy as np
import albumentations as A
import os
from skimage.io import imread
from dataclasses import dataclass
from typing import Tuple, Dict, List
# import sys; sys.path.append('..')
from data.cifar10 import load_train_val_data as load_cifar10_data
from data.HT_LIF_rawdata import load_HT_LIF_data
from data.data_location import DataLocation
from data.goprodehazing2017_data import get_train_val_test_data as load_gopro_data

def load_data(data_type, dataloc:DataLocation)->Dict[int, List[np.ndarray]]:
    if data_type == 'cifar10':
        return load_cifar10_data(dataloc.directory, [1,7])
    elif data_type in ['HT_LIF24', 'HT_T24', 'BioSR']:
        data_arr = load_HT_LIF_data(dataloc.directory)
        data_dict = {}
        for i in range(data_arr.shape[-1]):
            data_dict[i] = [x for x in data_arr[...,i]]
        return data_dict
    elif data_type in ['COSEM_jrc-hela', 'COSEM_jrc-choroid-plexus-2']:
        data = imread(dataloc.directory, plugin='tifffile')
        print('Read from', dataloc.directory, data.shape)
        data_dict = {}
        for i in range(data.shape[-1]):
            data_dict[i] = [x for x in data[...,i]]
        return data_dict
    elif data_type == 'goPro2017dehazing':
        return load_gopro_data(dataloc)
    else:
        assert data_type == "Hagen", "Only Hagen data is supported"
        if dataloc.fpath:
            return _load_data_fpath(dataloc.fpath)
        elif len(dataloc.channelwise_fpath) > 0:
            return _load_data_channelwise_fpath_hagen(dataloc.channelwise_fpath)

def compute_mean_stdev_based_normalization(data_dict, patch_size:int, numC:int, num_patches=10000):
    output = {
        'mean_channel': np.array([np.nan]*numC),
        'std_channel': np.array([np.nan]*numC),
    }
    for c_idx in range(numC):
        ch_data = data_dict[c_idx]
        mean_arr = []
        std_arr = []
        for _ in range(num_patches):
            idx = np.random.randint(0, len(ch_data))
            img = ch_data[idx]
            h,w = img.shape[-2:]
            h_idx = np.random.randint(0, h-patch_size)
            w_idx = np.random.randint(0, w-patch_size)
            patch = img[...,h_idx:h_idx+patch_size, w_idx:w_idx+patch_size]
            mean_arr.append(np.mean(patch))
            std_arr.append(np.std(patch))
        
        output['mean_channel'][c_idx] = np.mean(mean_arr)
        output['std_channel'][c_idx] = np.mean(std_arr)
    
    output['mean_input'] = np.array([np.nan])
    output['std_input'] = np.array([np.nan])

    return output

# def compute_normalization_dict(data_dict, channel_weights:List[float], numC:int, q_val=1.0, uint8_data=False):
#     """
#     x/x_max [0,1]
#     2 x/x_max -1 [-1,1]
#     (2x - x_max)/x_max [-1,1]
#     (x - x_max/2)/(x_max/2) [-1,1]
#     """
#     if uint8_data:
#         tar_max = 255
#         inp_max = tar_max * np.sum(channel_weights)
#         img_shape = data_dict[0][0].shape
#         tar1_max = tar_max
#         tar2_max = tar_max
#         if len(img_shape) == 2:
#             nC = 1
#         else:
#             nC = img_shape[0]
#         return {
#             'mean_input': inp_max/2,
#             'std_input': inp_max/2,
#             'mean_target': np.array([tar1_max/2]*nC + [tar2_max/2]*nC),
#             'std_target': np.array([tar1_max/2]*nC + [tar2_max/2]*nC),
#             # 
#             'ch0_max': tar1_max,
#             'ch1_max': tar2_max,
#             'input_max': inp_max
#         }

#     else:
#         mean_channel = []
#         std_channel = []
#         ch_unravel_list = []
#         output_dict = {}
#         for i in range(numC):
#             ch_unravel = np.concatenate([x.reshape(-1,) for x in data_dict[i]])
#             ch_max = np.quantile(ch_unravel, q_val)
#             mean_channel.append(ch_max/2)
#             std_channel.append(ch_max/2)
#             ch_unravel_list.append(ch_unravel)
#             output_dict[f'ch{i}_max'] = ch_max


#         inp_max = np.quantile(ch_unravel_list[0]*channel_weights[0]+(
#                               ch_unravel_list[1]*channel_weights[1]), 
#                               q_val)
        
#         output_dict.update({
#             'mean_input': inp_max/2,
#             'std_input': inp_max/2,
#             'mean_channel': np.array(mean_channel),
#             'std_channel': np.array(std_channel),
#             'input_max': inp_max
#         })
#         return output_dict


def _load_data_channelwise_fpath_hagen(fpaths:Tuple[str])-> Dict[int, List[np.ndarray]]:
    assert len(fpaths) == 2, "Only two channelwise fpaths are supported"
    data_ch0 = imread(fpaths[0], plugin='tifffile')
    data_ch1 = imread(fpaths[1], plugin='tifffile')
    print('HARDCODED upperclip to 1993. Disable it if not needed !!!')
    data_ch0[data_ch0 > 1993.0] = 1993.0
    data_ch1[data_ch1 > 1993.0] = 1993.0
    return {0: [x for x in data_ch0], 1: [x for x in data_ch1]}

def _load_data_fpath(fpath:str):
    assert fpath.exists(), f"Path {fpath} does not exist"
    assert os.splitext(fpath)[-1] == '.tif', "Only .tif files are supported"
    data = imread(fpath, plugin='tifffile')
    data_ch0 = data[...,0]
    data_ch1 = data[...,1]
    return {0: [x for x in data_ch0], 1: [x for x in data_ch1]}

class SplitDataset:
    def __init__(self, data_type, data_location:DataLocation, patch_size, 
                 target_channel_idx = None,
                 input_channel_idx=None,
                 random_patching=False, 
                 enable_transforms=False,
                 max_qval=0.98,
                 normalization_dict=None,
                 uncorrelated_channels=False,
                 channel_weights=None,
                 input_from_normalized_target=False,
                 real_input_fraction=None,
                 upper_clip=False,
                 normalize_channels=True):
        """
        Args:
        data_type: str - 'cifar10' or 'Hagen'
        data_location: DataLocation - location of the data (file path or directory)
        patch_size: int - size of the patch on which the model will be trained
        target_channel_idx: int - While the input is created from both channels, this decides which target needs to be predicted. If None, both channels are used as target.
        random_patching: bool - If True, random patching is done. Else, patches are extracted in a grid.
        enable_transforms: bool - If True, data augmentation is enabled.
        max_qval: float - quantile value for clipping the data and for computing the max value for the normalization dict.
        normalization_dict: dict - If provided, the normalization dict is used. Else, it is computed.
        uncorrelated_channels: bool - If True, the two diffrent random locations are used to crop patches from the two channels. Else, the same location is used.
        channel_weights: list - Input is the weighted sum of the two channels. If None, the weights are set to 1.
        upper_clip: bool - If True, the data is clipped to the max_qval quantile value.
        real_input_fraction: for what fraction of the dataset starting from index 0, should the real input be returned. Otherwise, all zero tensor is returned. 
        """
        # allowed_data_types = ['cifar10','Hagen', 'HT_LIF']
        # assert data_type in allowed_data_types, f"data_type must be one of {allowed_data_types}"

        self._patch_size = patch_size
        self._data_location = data_location
        self._channel_weights = channel_weights
        self._input_from_normalized_target = input_from_normalized_target
        self._normalize_channels = normalize_channels
        if self._channel_weights is None:
            self._channel_weights = [1,1]
        # channel_idx is the key. value is list of full sized frames.
        self._data_dict = self._load_data(data_type)
        self._numC = len(self._data_dict.keys())
        for i in range(self._numC):
            assert i in self._data_dict, f"Channel {i} has no data"
        
        self._frameN = min(len(self._data_dict[0]), len(self._data_dict[1]))
        self._target_channel_idx = target_channel_idx
        self._input_channel_idx = input_channel_idx
        self._real_input_fraction = real_input_fraction
        assert self._real_input_fraction is None or self._input_channel_idx is not None, "For real input fraction to make sense, we should have a real input"
        assert self._target_channel_idx is None or self._target_channel_idx <= self._numC, "target_channel_idx must be less than number of channels"
        assert self._input_channel_idx is None or self._input_channel_idx <= self._numC, "input_channel_idx must be less than number of channels"
        if self._real_input_fraction is not None:
            print(f'Using first {self.frames_with_real_input()}/{self._frameN} for real input')

        self._random_patching = random_patching
        self._uncorrelated_channels = uncorrelated_channels
        self._max_qval = max_qval

        self._transform = None
        if enable_transforms:
            self._transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
                ],
                additional_targets={f'image{k}': 'image' for k in range(2,2+self._numC-1)})

        if normalization_dict is None:
            print("Computing mean and std for normalization")
            normalization_dict = compute_mean_stdev_based_normalization(self._data_dict,  self._patch_size, self._numC)
            # normalization_dict = compute_normalization_dict(self._data_dict, self._channel_weights, self._numC, q_val=self._max_qval, uint8_data=data_type=='cifar10')
        
        self.normalization_dict = normalization_dict

        if upper_clip:
            print("Clipping data to {} quantile".format(self._max_qval))
            for ch_idx in self._data_dict.keys():
                self._data_dict[ch_idx] = [np.clip(x, 0, normalization_dict[f'ch{ch_idx}_max']) for x in self._data_dict[ch_idx]]

        assert 'mean_input' in normalization_dict, "mean_input must be provided"
        assert 'std_input' in normalization_dict, "std_input must be provided"
        assert 'mean_channel' in normalization_dict, "mean_channel must be provided"
        assert 'std_channel' in normalization_dict, "std_channel must be provided"

        msg = f'[{self.__class__.__name__}] Data: {self._frameN}x{len(self._data_dict.keys())}x{self._data_dict[0][0].shape}'
        msg += f' Patch:{patch_size} Random:{int(random_patching)} Aug:{self._transform is not None} Q:{self._max_qval}'
        if upper_clip is not None:
            msg += f' UpperClip:{int(upper_clip)}'
        msg += f'Uncor:{uncorrelated_channels}'
        if channel_weights is not None:
            msg += f' ChW:{self._channel_weights}'
        
        if self._real_input_fraction is not None:
            msg += f' RealFraction:{self._real_input_fraction}'

        if self._input_from_normalized_target:
            msg += f' InpFrmNormTar'
        

        print(msg)

    def _load_data(self, data_type):
        return load_data(data_type, self._data_location)

    def set_random_patching(self, boolean_flag):
        original = self._random_patching
        self._random_patching = boolean_flag
        print(f'Random patching set to {boolean_flag}')
        return original
    
    def get_input_target_normalization_dict(self):
        mean_input = self.normalization_dict['mean_input'].copy()
        std_input = self.normalization_dict['std_input'].copy()
        mean_channel = self.normalization_dict['mean_channel'].copy()
        std_channel = self.normalization_dict['std_channel'].copy()
        # input_max = self.normalization_dict['input_max'].copy()

        if self._input_channel_idx is not None:
            mean_input = mean_channel[self._input_channel_idx]
            std_input = std_channel[self._input_channel_idx]
            target_mask = np.ones(self._numC)
            target_mask[self._input_channel_idx] = 0
            mean_channel = mean_channel[target_mask.astype(bool)]
            std_channel = std_channel[target_mask.astype(bool)]
            # input_max = self.normalization_dict[f'ch{self._input_channel_idx}_max']

        output_dict ={
            'mean_input': mean_input.reshape(1,-1,1,1),
            'std_input': std_input.reshape(1,-1,1,1),
            'mean_channel': mean_channel.reshape(1,-1,1,1),
            'std_channel': std_channel.reshape(1,-1,1,1),
            # 'input_max': input_max,
        }
        # target_idx = 0
        # for ch_idx in range(self._numC):
        #     if ch_idx == self._input_channel_idx:
        #         continue
        #     output_dict[f'target{target_idx}_max'] = self.normalization_dict[f'ch{ch_idx}_max']
        #     target_idx += 1
        return output_dict

    def normalize_inp(self, inp):
        norm_inp = (inp - self.normalization_dict['mean_input'].reshape(-1,1,1))/self.normalization_dict['std_input'].reshape(-1,1,1)
        return norm_inp.astype(np.float32)
    
    def normalize_channels(self, channel_images):
        norm_tar = (channel_images - self.normalization_dict['mean_channel'].reshape(-1,1,1))/self.normalization_dict['std_channel'].reshape(-1,1,1)
        return norm_tar.astype(np.float32)
    
    def patch_count_per_frame(self):
        h,w = self._data_dict[0][0].shape[-2:]
        n_patches_per_frame = (h//self._patch_size) * (w//self._patch_size)
        return n_patches_per_frame
    
    def __len__(self):
        n_patches_per_frame = self.patch_count_per_frame()
        return self._frameN * n_patches_per_frame
    
    def frame_idx(self, index):
        return index // self.patch_count_per_frame()
    
    def patch_location(self, index):
        """
        Returns the frame index along with co-ordinates of the top-left corner of the patch in the frame.
        """
        frame_idx = self.frame_idx(index)
        index = index % self.patch_count_per_frame()
        h,w = self._data_dict[0][frame_idx].shape[-2:]
        h_idx = index // (w//self._patch_size)
        w_idx = index % (w//self._patch_size)
        return frame_idx, h_idx*self._patch_size, w_idx*self._patch_size


    def _get_location(self, index):
        if self._random_patching:
            frame_idx = np.random.randint(0, self._frameN)
            h,w = self._data_dict[0][frame_idx].shape[-2:]
            h_idx = np.random.randint(0, h-self._patch_size) if h > self._patch_size else 0
            w_idx = np.random.randint(0, w-self._patch_size) if w > self._patch_size else 0
        else:
            frame_idx, h_idx, w_idx = self.patch_location(index)
        return frame_idx, h_idx, w_idx
    
    def frames_with_real_input(self):
        return int(self._frameN*self._real_input_fraction)
    
    def __getitem__(self, index):

        frame_idx, h_idx, w_idx = self._get_location(index)
        img_list = []
        for i in range(self._numC):    
            img = self._data_dict[i][frame_idx]
            img_list.append(img)
        
        # img1 = self._data_dict[0][frame_idx]

        if self._uncorrelated_channels:
            for i in range(1, self._numC):
                frame_idx = np.random.randint(0, self._frameN)
                img_list[i] = self._data_dict[i][frame_idx]
        
        # img2 = self._data_dict[1][frame_idx]
        patch_arr = []
        for img in img_list:
            patch = img[...,h_idx:h_idx+self._patch_size, w_idx:w_idx+self._patch_size].astype(np.float32)
            patch_arr.append(patch)
        
        # random h,w location
        if self._transform:
            if patch_arr[0].ndim ==3:
                patch_arr = [x.transpose(1,2,0) for x in patch_arr]
            
            transform_kwargs = {f'image{k+1}': patch_arr[k] for k in range(1,self._numC)}
            transformed = self._transform(image=patch_arr[0], **transform_kwargs)
            patch_arr = [transformed['image']] + [transformed[f'image{k+1}'] for k in range(1,self._numC)]
            if patch_arr[0].ndim ==3:
                patch_arr = [x.transpose(2,0,1) for x in patch_arr]

        if patch_arr[0].ndim == 2:
            patch_arr = [x[None] for x in patch_arr]

        
        target = np.concatenate(patch_arr, axis=0)
        
        if self._normalize_channels:
            target = self.normalize_channels(target)
        
        real_input = None
        if self._input_channel_idx is not None and (self._real_input_fraction is not None and self._real_input_fraction > 0):
            if frame_idx <= self.frames_with_real_input():
                # for the initial real_input_fraction dataset, real input is returned.
                real_input = target[self._input_channel_idx:self._input_channel_idx+1]
            else:
                real_input = np.zeros_like(target[self._input_channel_idx:self._input_channel_idx+1])
        
        if self._input_channel_idx is not None:
            target_mask = np.ones(self._numC)
            target_mask[self._input_channel_idx] = 0
            target = target[target_mask.astype(bool)]

        assert self._target_channel_idx is None
        return {'target':target, 'input':real_input if real_input is not None else 0.0}
    

if __name__ == "__main__":
    import sys
    data_location = DataLocation(channelwise_fpath=('/group/jug/ashesh/data/diffsplit_hagen/val/val_actin-60x-noise2-highsnr.tif',
                                                    '/group/jug/ashesh/data/diffsplit_hagen/val/val_mito-60x-noise2-highsnr.tif'))
    # patch_size = 512
    # data_type = 'hagen'
    # data_location = DataLocation(directory='/group/jug/ashesh/data/cifar-10-python/train')
    patch_size = 256
    data_type = 'Hagen'
    nC = 1 if data_type == 'Hagen' else 3
    uncorrelated_channels = False
    channel_weights = [1,0.3]
    dataset = SplitDataset(data_type, data_location, patch_size, 
                                max_qval=0.98, upper_clip=True,
                             normalization_dict=None, enable_transforms=True,
                             channel_weights=channel_weights,
                             uncorrelated_channels=True, random_patching=True,
                             input_from_normalized_target=True)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        inp = data['input']
        target = data['target']
        print(inp.min(), inp.max(),end='\t')
        print(target[0].min(), target[0].max(), end='\t')
        print(target[1].min(), target[1].max())


    import matplotlib.pyplot as plt
    data= dataset[0]
    inp = data['input']
    target = data['target']
    _,ax = plt.subplots(figsize=(6,2),ncols=3)
    ax[0].imshow((2+inp.transpose(1,2,0))/4)
    ax[1].imshow((1 +target[:nC].transpose(1,2,0))/2)
    ax[2].imshow((1+target[nC:].transpose(1,2,0))/2)
    # disable axis
    for a in ax:
        a.axis('off')