import os
from skimage.io import imread

def load_HT_LIF_data(directory):
    fnames = [os.path.join(directory, f) for f in os.listdir(directory)]
    assert len(fnames) == 1
    fpath  = os.path.join(directory, fnames[0])
    print('Loading data from', fpath)
    return imread(fpath, plugin='tifffile')
