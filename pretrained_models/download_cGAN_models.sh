
mkdir -p ./checkpoints/GlyphNet_pretrain
mkdir -p ./checkpoints/OrnaNet_pretrain

MODEL_FILE=./checkpoints/GlyphNet_pretrain/400_net_G.pth
URL=https://people.eecs.berkeley.edu/~sazadi/MCGAN/models/400_net_G.pth
wget -N $URL -O $MODEL_FILE

MODEL_FILE=./checkpoints/GlyphNet_pretrain/400_net_G_3d.pth
URL=https://people.eecs.berkeley.edu/~sazadi/MCGAN/models/400_net_G_3d.pth
wget -N $URL -O $MODEL_FILE

MODEL_FILE=./checkpoints/OrnaNet_pretrain/100_net_G1_pretrained.pth
URL=https://people.eecs.berkeley.edu/~sazadi/MCGAN/models/100_net_G_gray2rgb.pth
wget -N $URL -O $MODEL_FILE

MODEL_FILE=./checkpoints/OrnaNet_pretrain/100_net_D1_pretrained.pth
URL=https://people.eecs.berkeley.edu/~sazadi/MCGAN/models/100_net_D_gray2rgb.pth

wget -N $URL -O $MODEL_FILE
