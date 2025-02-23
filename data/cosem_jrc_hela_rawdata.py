from skimage.io import imread
import numpy as np

def load_jrc_hela_rawdata(input_fpath):
    ch1_fpath = input_fpath.replace('_bleedthrough_', '_ch1_')
    ch2_fpath = input_fpath.replace('_bleedthrough_', '_ch2_')
    # Load the images
    ch1_stack = imread(ch1_fpath, plugin='tifffile')
    ch2_stack = imread(ch2_fpath, plugin='tifffile')
    inp_stack = imread(input_fpath, plugin='tifffile')
    print(inp_stack.shape, ch1_stack.shape, ch2_stack.shape)
    data_stack = np.stack([ch1_stack, ch2_stack, inp_stack[:,0], inp_stack[:,1]], axis=-1)
    return data_stack