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
 
# =======================================
##COPY pretrained network from its corresponding directory
# =======================================

model_1_pretrained="./checkpoints/GlyphNet_pretrain" 
if [ ! -f "./checkpoints/${experiment_dir}/400_net_G.pth" ]; then
    cp "${model_1_pretrained}/400_net_G.pth" "./checkpoints/${experiment_dir}/"
    cp "${model_1_pretrained}/400_net_G_3d.pth" "./checkpoints/${experiment_dir}/"
fi

 

# =======================================
## Make a video by testing model on different training epochs
# =======================================
all_epochs=()
init_epoch=$(seq 0 1 25);
mid_epoch=$(seq 26 2 100 );
rest_epoch=$(seq 101 20 $EPOCH1 );
all_epochs+=$init_epoch
all_epochs+=" "
all_epochs+=$mid_epoch
all_epochs+=" "
all_epochs+=$rest_epoch

echo $all_epochs
for i in $all_epochs;

	do CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_Stack.py --dataroot ${DATASET} --name "${experiment_dir}" --model ${MODEL}\
										 --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --norm ${NORM}\
										 --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1}\
										 --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --align_data\
										 --loadSize ${LOADSIZE} --display_id 0 --batchSize 1 --conditional --rgb_out --partial\
										 --which_epoch $EPOCH --which_epoch1 $i --blanks 0 --conv3d --base_root ${base_dir}
done

CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_video.py --dataroot ${DATASET} --name "${experiment_dir}" --model ${MODEL} \
								--which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --norm ${NORM}\
								--input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1} --which_model_preNet ${PRENET} \
							    --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --align_data  --display_id 0\
								--batchSize 1 --conditional --rgb_out --partial --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1} --blanks 0 --conv3d --base_root ${base_dir}



