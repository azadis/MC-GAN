################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################

from torch import transpose
import torch.utils.data as data
from torch import index_select,LongTensor
from PIL import Image
import os
import os.path
import numpy as np
from scipy import misc
import random
from options.train_options import TrainOptions
import torch

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


def font_transform(img,path, rgb_in):
    n_rgb = img.size()[0]
    target_size = img.size()[1]
    D_ = img.size()[2]/target_size
    # warnings.warn("size, %s %s"%(img.size(),D_))
    if not rgb_in:
        img = torch.mean(img,dim=0) #only one of the RGB channels    
        img = img[None,:,:] #(1,64,64)
        n_rgb =1
    else:
        img = img.permute(1,0,2).contiguous().view(1,target_size, n_rgb*img.size()[2])
        
    slices = []
    for j in range(target_size):
        for i in np.arange(0,D_):
            slices += list(target_size * np.arange(i,D_*n_rgb,26) + j)
    img = index_select(img,2,LongTensor(slices)).view(target_size,target_size,D_*n_rgb)
    img = img.permute(2,0,1)
    return img           


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, rgb=False, fineSize=0,
                 loadSize=0,font_trans=False,no_permutation=False):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        
        if no_permutation:
           self.imgs= sorted(self.imgs)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.rgb = rgb
        self.font_trans = font_trans
        self.img_crop = [False]*len(imgs) #whole image to be included
        self.fineSize=fineSize
        self.loadSize=loadSize


   
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
			img = self.transform(img)
			if (self.font_trans):
				img = font_transform(img,path, self.rgb)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
