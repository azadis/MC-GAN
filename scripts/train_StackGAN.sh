#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test End-to-End network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================

DATA=$1
DATASET="../datasets/public_web_fonts/${DATA}/"
experiment_dir="${DATA}_MCGAN_train"
base_dir="../datasets/Capitals64/BASE"
NAME="${experiment_dir}"
MODEL=StackGAN
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
IN_NC=26
O_NC=26
IN_NC_1=3
O_NC_1=3
PRENET=2_layers
LR=0.002
FINESIZE=64
LOADSIZE=64
LAM_A=300
LAM_C=10
NITER=400
NITERD=300
BATCHSIZE=7
EPOCH=400
EPOCH1=$(($NITER+$NITERD))
CUDA_ID=1


if [ ! -d "./checkpoints/${experiment_dir}" ]; then
    mkdir "./checkpoints/${experiment_dir}"
fi
LOG="./checkpoints/${experiment_dir}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi

# =======================================
##COPY pretrained network from its corresponding directory
# =======================================
model_1_pretrained="./checkpoints/GlyphNet_pretrain" 
if [ ! -f "./checkpoints/${experiment_dir}/400_net_G.pth" ]; then
    cp "${model_1_pretrained}/400_net_G.pth" "./checkpoints/${experiment_dir}/"
    cp "${model_1_pretrained}/400_net_G_3d.pth" "./checkpoints/${experiment_dir}/"
fi


exec &> >(tee -a "$LOG")

# =======================================
## Train End-2-End model
# =======================================
echo "TRAIN MODEL WITH REAL TRAINING DATA" 

CUDA_VISIBLE_DEVICES=${CUDA_ID} python train_Stack.py --dataroot ${DATASET}  --name ${NAME} --model ${MODEL}\
							  --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} \
							  --norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1}\
							  --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --lambda_A ${LAM_A}\
							  --lambda_C ${LAM_C} --align_data --use_dropout --display_id 0 --niter ${NITER} --niter_decay ${NITERD}\
							  --batchSize ${BATCHSIZE} --conditional --save_epoch_freq 100 --rgb_out --partial --which_epoch ${EPOCH} \
							  --display_freq 5 --print_freq 5 --blanks 0 --conv3d --base_font --base_root ${base_dir} #--gpu_ids 0,1 


# =======================================
## BASELINE: train only the second network on top of clean b/w glyphs
# =======================================

# CUDA_VISIBLE_DEVICES=${CUDA_ID} python train_Stack.py --dataroot ${DATASET}  --name ${NAME} --model ${MODEL}  
                                # --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --norm ${NORM}
                                # --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1} 
                                # --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --lambda_A ${LAM_A} 
                                # --lambda_C ${LAM_C} --align_data --use_dropout --display_id 0 --niter ${NITER} --niter_decay ${NITERD} 
                                # --batchSize ${BATCHSIZE} --conditional --save_epoch_freq 100 --rgb_out --partial --which_epoch ${EPOCH}
                                # --display_freq 5 --print_freq 5 --blanks 0 --base_font --conv3d --no_Style2Glyph --orna










