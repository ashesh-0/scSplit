from data.split_dataset import SplitDataset, DataLocation
import numpy as np

class RestorationDataset(SplitDataset):
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
                 normalize_channels=True,
                 mix_target_max_factor=0.5):
        super().__init__(data_type, data_location, patch_size, 
                 target_channel_idx=target_channel_idx,
                 input_channel_idx=input_channel_idx,
                 random_patching=random_patching, 
                 enable_transforms=enable_transforms,
                 max_qval=max_qval,
                 normalization_dict=normalization_dict,
                 uncorrelated_channels=uncorrelated_channels,
                 channel_weights=channel_weights,
                 input_from_normalized_target=input_from_normalized_target,
                 real_input_fraction=real_input_fraction,
                 upper_clip=upper_clip,
                 normalize_channels=normalize_channels)
        self._mix_target_max_factor = mix_target_max_factor
        # do we want to mix the target with the input to create the input?
        assert self._mix_target_max_factor >= 0 and self._mix_target_max_factor <= 1, "mix_target_max_factor must be between 0 and 1"
        if self._mix_target_max_factor > 0:
            print(f'[{self.__class__.__name__}] {self._mix_target_max_factor} of target will be mixed with input')


    def get_input_target_normalization_dict(self):
        mean_input = self.normalization_dict['mean_input'].copy()
        std_input = self.normalization_dict['std_input'].copy()
        mean_target, mean_input = self.normalization_dict['mean_channel'].copy()
        std_target, std_input = self.normalization_dict['std_channel'].copy()
        output_dict ={
            'mean_input': np.array(mean_input).reshape(1,-1,1,1),
            'std_input': np.array(std_input).reshape(1,-1,1,1),
            'mean_channel': np.array(mean_target).reshape(1,-1,1,1),
            'std_channel': np.array(std_target).reshape(1,-1,1,1),
            # 'input_max': input_max,
        }
        return output_dict
    
    # def patch_location(self, index):
    #     loc = super().patch_location(index)
    #     return (loc[0],0,*loc[1:])
    
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

        assert len(patch_arr) == 2, "Input & target must be a pair"
        output_dict = {'target': patch_arr[0], 'input': patch_arr[1]}
        if self._mix_target_max_factor>0:
            factor = np.random.rand()*self._mix_target_max_factor
            output_dict['input'] = output_dict['input'] * (1-factor) + output_dict['target'] * factor
            output_dict['target_factor'] = factor
        
        return output_dict

    