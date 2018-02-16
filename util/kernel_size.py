################################################################################
# MC-GAN
# Compute size of output feature map given dim, stride, padding, dilation 
# By Samaneh Azadi
################################################################################


import sys
import numpy as np


def conv2d(input_dim, padding, kernel_size, stride, dilation):
    output_dim = np.floor((input_dim + 2*padding - dilation*(kernel_size-1) -1)/stride +1 )
    return output_dim        

def convTranspose2d(input_dim, padding, kernel_size, stride, output_padding):
    output_dim = (input_dim-1)*stride - 2*padding + kernel_size + output_padding
    return output_dim        


    
if __name__ == '__main__':
    conv_type = str(sys.argv[1])
    input_dim = float(sys.argv[2])
    padding = float(sys.argv[3])
    kernel_size = float(sys.argv[4])
    stride = float(sys.argv[5])
    if conv_type == 'downconv':        
        if len(sys.argv) == 7:
           dilation = float(sys.argv[6])
        else:
           dilation = 1
        output_dim = conv2d(input_dim, padding, kernel_size, stride, dilation)
        print "kernel of dim (%s,%s) with stride %s, padding %s, dilation %s has been applied on the input with dim (%s,%s)"%(kernel_size, kernel_size,stride,padding,dilation, input_dim, input_dim)

    elif conv_type == 'upconv':
        if len(sys.argv) == 7:
           output_padding = float(sys.argv[6])
        else:
           output_padding = 0
        output_dim = convTranspose2d(input_dim, padding, kernel_size, stride, output_padding)
        print "kernel of dim (%s,%s) with stride %s, padding %s, dilation %s has been applied on the input with dim (%s,%s)"%(kernel_size, kernel_size,stride,padding,output_padding, input_dim, input_dim)
    print " dim of the output feature map is ", output_dim