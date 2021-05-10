from net import *
from utils import *
from Logger import *
from trans_graph import *

import time
import datetime
import sys
import gc
import math
import os
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import LocalOutlierFactor


class LOF(object):
    def __init__(self, args):
        self.model_name = 'LOF'
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.dataset_name = args.dataset

        self.train_len = args.dataset_setting[self.dataset_name][0]
        self.test_len = args.dataset_setting[self.dataset_name][1]
        self.num_sensor = args.dataset_setting[self.dataset_name][2]
        self.num_sensor_dev = args.dataset_setting[self.dataset_name][3]
        self.new_start = args.new_start

        self.epoch = args.epoch
        # self.iteration = args.iteration##
        self.resume_iters = args.resume_iters
        self.dropout = args.dropout

        self.loss_function = eval(args.loss_function)
        self.rx_w = args.rx_w
        self.nx_w = args.nx_w
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch
        self.init_lr = args.lr

        self.print_net = args.print_net

        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.use_tensorboard = args.use_tensorboard

        self.batch_size = args.batch_size
        self.num_clips = args.num_clips
        self.embedding_dim = args.dataset_setting[self.dataset_name][4]
        self.graph_ch = args.dataset_setting[self.dataset_name][5]
        self.conv_ch = args.dataset_setting[self.dataset_name][6] if args.conv_ch == 0 else args.conv_ch
        self.original_dim = args.dataset_setting[self.dataset_name][7]
        self.head = args.head

        # build graph
        print(" [*] Buliding model!")
        self.build_model()

        print("##### Information #####")
        print("# loss function: ", args.loss_function)
        print("# dataset : ", self.dataset_name)
        # print("# batch_size : ", self.batch_size)


    def build_model(self):
        self.G = LocalOutlierFactor()

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def test(self):
        node_feature = []
        abnormal = []

        for d in range(self.test_len):
            node_path = self.dataset_name + '/node/testnode' + str(d + 1) + '.npy'
            edge_path = self.dataset_name + '/graph/testgraph' + str(d + 1) + '.npy'
            ab_path = self.dataset_name + '/abnormal/abnormal' + str(d + 1) + '.npy'
            node, _, _, ab = load_graph(node_path, edge_path,abnormal_path=ab_path)

            # graph level
            # node_feature.append(node.view(1,-1))
            # abnormal.append(int(torch.sum(ab)>0))

            # node level
            node_feature.append(node)
            abnormal.append(ab.view(-1).numpy())

        node_feature = torch.cat(node_feature).numpy()
        label = np.array(abnormal).flatten()

        loss = {}

        # =================================================================================== #
        #                             2. Train the Model                                      #
        # =================================================================================== #
        x = self.G.fit_predict(node_feature)

        record1 = (x*(-1) + 1) / 2


        print("Finish LOF Part!")


        tp = (label * record1).sum()
        tn = ((1 - label) * (1 - record1)).sum()
        fp = ((1 - label) * record1).sum()
        fn = (label * (1 - record1)).sum()
        print(tp,tn,fp,fn)

        epsilon = 1e-7

        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn + epsilon)
        prec = tp / (tp + fp + epsilon)
        f1 = 2 * (recall * prec) / (recall + prec)

        print('Best result for now: f1={}, acc={}, recall={}, prec={}'.format(f1, acc, recall, prec))