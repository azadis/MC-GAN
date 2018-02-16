################################################################################
# MC-GAN
# plot different loss values of StackGAN_model during training from the log file
# By Samaneh Azadi
################################################################################


import os
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='check how training has gone so far')
    parser.add_argument('--logRoot', required=True, help='path to the directory containing output log')
    parser.add_argument('--avg', default=100, required=False, help='number of points to take average over')

    return parser.parse_args()


def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
    

def plot_loss(loss_path,lossType, col,n_avg):
    file_ =open('%s/output.txt'%loss_path)
    lines = file_.readlines()
    lines = [line.split("%s: "%lossType) for line in lines]
    loss=[]
    epochs = []
    for line in lines:
        if len(line)==1:
            continue
        else:
            loss.append(float(line[1].split(" ")[0].strip())) 
            epochs.append(float(line[0].split("epoch: ")[1].split(",")[0].strip()))
    loss_avg = moving_average(loss,int(n_avg))
    return loss_avg,epochs


def main():
    opt = parse_args()
    letter_2_id = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,
               'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,
               'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,
               'W':22,'X':23,'Y':24,'Z':25}


    loss2col = {}
    loss2col['G1_GAN'] = 8
    loss2col['G1_L1'] = 10
    loss2col['G1_MSE_gt'] = 12
    loss2col['G1_MSE'] = 14
    loss2col['D1_real'] = 16
    loss2col['D1_fake'] = 18
    loss2col['G_L1'] = 20

    loss_path = opt.logRoot
    n_avg = opt.avg

    ind = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]
    loss_type = ['G1_GAN','G1_L1','G1_MSE_gt','G1_MSE','D1_real','D1_fake','G_L1']
    loss_ = []
    epoch_ = []
    fig, axes = plt.subplots(2,4, figsize=(14,8))
    for i in range(len(loss_type)):
        lossType = loss_type[i]
        print lossType
        loss,epoch = plot_loss(loss_path, lossType,loss2col[lossType],n_avg)
        loss_.append(np.array(loss))
        epoch_.append(np.array(epoch))
        axes[ind[i][0],ind[i][1]].plot(loss)
        axes[ind[i][0],ind[i][1]].set_title(lossType)
    print 'save to %s/losses.png'%(loss_path)
    plt.savefig('%s/losses.png'%loss_path)
    
    
if __name__ == '__main__':
    main()
 