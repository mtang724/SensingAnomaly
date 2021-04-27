#import tensorflow as tf
#from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

##################################################################################
# Loss function
##################################################################################


def l1_loss(x, y, graph=True):
    if graph:
        loss = torch.mean(torch.abs(x - y))
    else:
        loss = torch.mean(torch.abs(x - y), dim=-1)

    return loss.to(DEVICE)


def l2_loss(x, y, graph=True, c=0):
    if graph:
        loss = torch.clamp(torch.mean(torch.pow(x-y, 2)) - c, min=0.)
    else:
        loss = torch.clamp(torch.mean(torch.pow(x-y, 2), dim=-1) - c, min=0.)

    return loss.to(DEVICE)


def cross_entropy(output, lable, graph=True):
    if graph:
        loss = torch.mean(-1 * lable * torch.log(output + 1e-15) - (1-lable) * torch.log(1-output + 1e-15))
    else:
        loss = torch.mean(-1 * lable * torch.log(output + 1e-15) - (1-lable) * torch.log(1-output + 1e-15), dim=-1)

    return  loss.to(DEVICE)

