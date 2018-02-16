import time
import os
from options.test_options import TestOptions
import numpy as np
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle
opt.stack = True
opt.use_dropout = False
opt.use_dropout1 = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

epoch_list = list(range(26))+list(np.arange(26,101,2))+list(np.arange(101,int(opt.which_epoch1)+1,20))
for epoch in epoch_list:
	opt.which_epoch1 = epoch
	model = create_model(opt)
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch+'+'+str(epoch)))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch+'+'+str(epoch)))
	# test
	for i, data in enumerate(dataset):
	    if i >= opt.how_many:
	        break
	    model.set_input(data)
	    model.test()
	    visuals = model.get_current_visuals()
	    img_path = model.get_image_paths()
	    print('process image... %s' % img_path)
	    visualizer.save_images(webpage, visuals, img_path)

	webpage.save()

video_path = os.path.join(opt.results_dir, opt.name, '%s' % (opt.phase))
print "save to:%s"%video_path
if not os.path.isdir(video_path):
	os.mkdir(video_path)
visualizer.save_video(video_path)

