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
base_dir="../datasets/Capitals64/BASE"
experiment_dir="${DATA}_MCGAN_train"
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
FINESIZE=64
LOADSIZE=64
BATCHSIZE=1
EPOCH1=700
EPOCH=400
CUDA_ID=0


if [ ! -d "./checkpoints/${experiment_dir}" ]; then
	mkdir "./checkpoints/${experiment_dir}"
fi
LOG="./checkpoints/${experiment_dir}/test.txt"
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
## Test End-2-End model
# =======================================
CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_Stack.py --dataroot ${DATASET} --name "${experiment_dir}" --model ${MODEL}\
								 --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D}\
								 --norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1}\
								 --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --display_id 0\
								 --batchSize 1 --conditional --rgb_out --partial --align_data --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1}\
								 --blanks 0 --conv3d  --base_root ${base_dir} 



# =======================================
## test only the second network for clean b/w glyphs
# =======================================

# CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_Stack.py --dataroot ${DATASET} --name "${experiment_dir}" --model ${MODEL}\
								 # --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --norm ${NORM} \
								 # --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1} --which_model_preNet ${PRENET}\
								 #  --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --align_data --display_id 0 \
								 #  --batchSize 1 --conditional --rgb_out --partial  --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1} --blanks 0\
								 #   --conv3d --no_Style2Glyph --orna --base_root ${base_dir}




