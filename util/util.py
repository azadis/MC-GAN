from __future__ import print_function
import torch
import numpy as np
from PIL import Image, ImageOps
import inspect, re
import numpy as np
import os
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def HorizontalFlip(img):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5.
    Args:
       img (PIL.Image): Image to be flipped.
    Returns:
		PIL.Image: flipped image.
    """
    return ImageOps.mirror(img)

def VerticalFlip(img):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5.
	Args:
		img (PIL.Image): Image to be flipped.
	Returns:
		PIL.Image: flipped image.
	"""
    return ImageOps.flip(img)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path,rgb):
    n_ch = image_numpy.shape[2]
    if n_ch==1:
        image_numpy = np.squeeze(image_numpy)
    elif n_ch>3:
        if not rgb:
            image_numpy = np.reshape(image_numpy,(image_numpy.shape[0], image_numpy.shape[1]*n_ch),order='F')
        else:
            
            image_numpy_ = np.zeros((image_numpy.shape[0],image_numpy.shape[1]*n_ch/3,3))
            for i in range(n_ch/3):
                image_numpy_[:,i*image_numpy.shape[1]:(i+1)*image_numpy.shape[1],:]= image_numpy[:,:,i*3:(i+1)*3]
            image_numpy = image_numpy_.astype('uint8')
                 
    
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
