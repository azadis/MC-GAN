################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################

import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from models.models import create_model
from util.visualizer import Visualizer
from data.data_loader import CreateDataLoader

opt.stack = True
data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
opt.use_dropout = False
opt.use_dropout1 = True
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0
  
epoch =int(opt.which_epoch1)
epoch0 = epoch 
print "starting propagating back to the first network with starting lr %s ..."%opt.lr
opt.lr = opt.lr
opt.continue_train = False
opt.use_dropout = True
opt.use_dropout1 = True
model = create_model(opt)
visualizer = Visualizer(opt) 
print('saving the model at the end of epoch %d, iters %d' %
    (epoch0, total_steps))
model.save(epoch0)

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        if not opt.no_Style2Glyph:
            model.optimize_parameters_Stacked(epoch)
        else:
            model.optimize_parameters(epoch)

    
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch+epoch0, total_steps))
            model.save('latest')

    if (epoch % opt.save_epoch_freq == 0):# or (epoch<20):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch+epoch0, total_steps))
        model.save('latest')
        model.save(epoch+epoch0)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
