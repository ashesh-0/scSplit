import os
import numpy as np
from PIL import Image
from data.data_location import DataLocation
from tqdm import tqdm

def get_train_val_fpaths(rootdir, val_ratio=0.1):
    fpaths = get_paths(os.path.join(rootdir, 'train'))
    np.random.RandomState(955).shuffle(fpaths)
    val_len = int(len(fpaths) * val_ratio)
    val_fpaths = fpaths[:val_len]
    train_fpaths = fpaths[val_len:]
    return train_fpaths, val_fpaths


def get_paths(rootdir):
    subdirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]
    blur_fpaths = []
    for subdir in subdirs:
        databucket = os.path.join(rootdir, subdir)
        fnames = os.listdir(os.path.join(databucket, 'blur'))
        blur_fpaths += [os.path.join(databucket, 'blur', f) for f in fnames]
    
    sharp_fpaths = [x.replace('/blur/', '/sharp/') for x in blur_fpaths]

    zip_fpaths = list(zip(blur_fpaths, sharp_fpaths))
    return zip_fpaths


def get_test_fpaths(rootdir):
    return get_paths(os.path.join(rootdir, 'test'))

def load_data_dict(fpaths, limit_count=None):
    if limit_count is not None:
        print('---------------------------------------------------')
        print(f'Loading limited number of images :{limit_count}!!')
        print('---------------------------------------------------')  
    
    blur_imgs = []
    sharp_imgs = []
    i = 0
    for blur_fpath, sharp_fpath in tqdm(fpaths):
        img = Image.open(blur_fpath)
        blur_imgs.append(np.array(img).transpose((2,0,1)))
        img = Image.open(sharp_fpath)
        sharp_imgs.append(np.array(img).transpose((2,0,1)))
        i +=1
        if limit_count is not None and i > limit_count:
            break
        # if i > 20:
        #     break
    return {0: sharp_imgs, 1: blur_imgs}

def get_train_val_test_data(datalocation:DataLocation):
    rootdir = datalocation.directory
    datasplit_type = datalocation.datasplit_type
    if datasplit_type in ['train','val']:
        train_fpaths, val_fpaths = get_train_val_fpaths(rootdir, val_ratio=0.1)
        if datasplit_type == 'train':
            print('Loading train data')
            fpaths = train_fpaths
        else:
            print('Loading val data')
            fpaths = val_fpaths
    elif datasplit_type == 'test':
        print('Loading test data')
        fpaths = get_test_fpaths(rootdir)
    else:
        raise ValueError(f"Unknown datasplit type: {datasplit_type}")
    
    return load_data_dict(fpaths, limit_count=datalocation.limit_count)


if __name__ == '__main__':
    dataloc = DataLocation(directory='/group/jug/ashesh/data/goproDeblurring2017/GOPRO_Large', datasplit_type='train')
    data_dict = get_train_val_test_data(dataloc)
    breakpoint()