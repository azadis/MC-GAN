#=====================================
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
#=====================================


import time
import os
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
ssim_score = 0
mse_score = 0
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    scores = visualizer.eval_current_result(visuals)
    print "ssim: %s"%(scores[0])
    print "MSE: %s"%(scores[1])
    ssim_score += scores[0]
    mse_score += scores[1]
    visualizer.save_images(webpage, visuals, img_path)

print("Final SSIM score & MSE score for %s images:"%(i+1), ssim_score/(i+1), mse_score/(i+1))
webpage.save()