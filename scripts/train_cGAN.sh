#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test conditional GAN Glyph network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================

DATA=$1
DATASET="../datasets/${DATA}/"
experiment_dir="GlyphNet_pretrain"
MODEL=cGAN
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
IN_NC=26
O_NC=26
GRP=26
PRENET=2_layers
FINESIZE=64
LOADSIZE=64
LAM_A=100
NITER=500
NITERD=100
BATCHSIZE=150
CUDA_ID=1


if [ ! -d "./checkpoints/${experiment_dir}" ]; then
	mkdir "./checkpoints/${experiment_dir}"
fi
LOG="./checkpoints/${experiment_dir}/output.txt"
if [ -f $LOG ]; then
	rm $LOG
fi
 


exec &> >(tee -a "$LOG")

# =======================================
## Train Glyph Network on font dataset
# =======================================
CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py --dataroot ${DATASET} --name "${experiment_dir}" \
								--model ${MODEL} --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D}  --n_layers_D ${n_layers_D} --which_model_preNet ${PRENET}\
								--norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --grps ${GRP} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --lambda_A ${LAM_A} --align_data --use_dropout\
								--display_id 0 --niter ${NITER} --niter_decay ${NITERD} --batchSize ${BATCHSIZE} --conditional --save_epoch_freq 100 --print_freq 100 --conv3d 


# =======================================
## Train on RGB inputs to generate RGB outputs; Image Translation in the paper
# =======================================
# CUDA_VISIBLE_DEVICES=2 python ~/AdobeFontDropper/train.py --dataroot ../datasets/Capitals_colorGrad64/ --name "${experiment_dir}"\
						 # --model cGAN --which_model_netG resnet_6blocks --which_model_netD n_layers --n_layers_D 1 --which_model_preNet 2_layers \
						 # --norm batch --input_nc 78 --output_nc 78 --fineSize 64 --loadSize 64 --lambda_A 100 --align_data --use_dropout \
						 # --display_id 0 --niter 500 --niter_decay 1000 --batchSize 100 --conditional --save_epoch_freq 20 --display_freq 2 --rgb

# =======================================
## Consider input as tiling of input glyphs rather than a stack
# =======================================

# CUDA_VISIBLE_DEVICES=2 python ~/AdobeFontDropper/train.py --dataroot ../datasets/Capitals64/ --name "${experiment_dir}" \
				# --model cGAN --which_model_netG resnet_6blocks --which_model_netD n_layers  --n_layers_D 1 --which_model_preNet 2_layers\
				# --norm batch --input_nc 1 --output_nc 1 --fineSize 64 --loadSize 64 --lambda_A 100 --align_data --use_dropout\
				# --display_id 0 --niter 500 --niter_decay 2000 --batchSize 5 --conditional --save_epoch_freq 10 --display_freq 5 --print_freq 100 --flat



