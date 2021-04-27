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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DGAD(object):
    def __init__(self, args):
        self.model_name = 'Graph_Conv_AE'
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.dataset_name = args.dataset

        self.train_len = args.dataset_setting[self.dataset_name][0]
        self.test_len = args.dataset_setting[self.dataset_name][1]
        self.num_sensor = args.dataset_setting[self.dataset_name][2]
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
        self.embedding_dim = args.dataset_setting[self.dataset_name][3]
        self.graph_ch = args.dataset_setting[self.dataset_name][4]
        self.conv_ch = args.dataset_setting[self.dataset_name][5] if args.conv_ch == 0 else args.conv_ch
        self.original_dim = args.dataset_setting[self.dataset_name][6]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # build graph
        print(" [*] Buliding model!")
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        print("##### Information #####")
        print("# loss function: ", args.loss_function)
        print("# dataset : ", self.dataset_name)
        # print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        # print("# iteration per epoch : ", self.iteration)

        # torch.autograd.set_detect_anomaly(True)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        if self.resume_iters:
            checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            print('Loading the trained models from step {}...'.format(resume_iters))
            G_path = os.path.join(checkpoint_dir, '{}-G.ckpt'.format(resume_iters))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def save(self, save_dir, counter):
        self.model_save_dir = os.path.join(save_dir, self.model_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(counter + 1))
        torch.save(self.G.state_dict(), G_path)

        print('Saved model {} checkpoints into {}...'.format(counter + 1, self.model_save_dir))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def build_model(self):
        self.G = Detector(self.graph_ch, self.conv_ch, self.num_sensor, self.embedding_dim, self.dropout,
                          self.original_dim)

        if self.print_net:
            self.print_network(self.G, 'G')

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.init_lr)

        self.G.to(self.device)

    def train(self):
        start_iters = self.resume_iters if not self.new_start else 0
        self.restore_model(self.resume_iters)

        self.iteration = self.train_len - self.num_clips + 1

        start_epoch = (int)(start_iters / self.iteration)
        start_batch_id = start_iters - start_epoch * self.iteration

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr

        self.set_requires_grad([self.G], True)

        self.G.train()

        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag and epoch > self.decay_epoch:
                lr = self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)  # linear decay
                self.update_lr(lr)

            for idx in range(start_batch_id, self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data(Unfinisher)                    #
                # =================================================================================== #
                node_feature = torch.zeros((self.num_clips, self.num_sensor, self.graph_ch), dtype=torch.float)
                edge = torch.zeros((self.num_clips, self.num_sensor, self.num_sensor), dtype=torch.float)

                edge = edge.to(self.device)
                node_feature = node_feature.to(self.device)

                loss = {}

                # =================================================================================== #
                #                             2. Train the Model                                      #
                # =================================================================================== #
                recon, forecast, node_recon = self.G(node_feature, edge)

                self.recon_error = self.loss_function(recon, node_feature[-1])
                self.forecast_error = self.loss_function(forecast, node_feature[-1])

                self.Graph_error = (self.rx_w * self.recon_error
                                    + (1 - self.rx_w) * self.forecast_error)
                self.Node_error = self.loss_function(node_recon, node_feature[-1])

                self.Error = (self.nx_w * self.Node_error
                              + (1 - self.rx_w) * self.Graph_error)

                # Logging.
                loss['reconstruction_error'] = self.recon_error.item()
                loss['Forecast_error'] = self.forecast_error.item()
                loss['Graph_error'] = self.Node_error.item()
                loss['Node_error'] = self.Graph_error.item()
                loss['Whole_error'] = self.Error.item()

                del recon
                del forecast
                del node_recon
                torch.cuda.empty_cache()

                self.reset_grad()
                self.Error.backward()
                self.g_optimizer.step()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #
                start_iters += 1

                # Print out training information.
                if idx % self.print_freq == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]".format(et, epoch + 1, self.epoch, idx + 1,
                                                                                  self.iteration)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                        if self.use_tensorboard:
                            self.logger.scalar_summary(tag, value, start_iters)
                    print(log)

                # Save model checkpoints.
                if (idx + 1) % self.save_freq == 0:
                    self.save(self.checkpoint_dir, start_iters)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, start_iters)

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def test(self):
        self.restore_model(self.resume_iters)

        self.G.eval()
        self.set_requires_grad(self.G, False)

        self.iteration = self.test_len - self.num_clips + 1
        self.model_save_dir = os.path.join(self.checkpoint_dir, self.model_dir)

        with torch.no_grad():
            for idx in range(self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data(Unfinished)                    #
                # =================================================================================== #
                node_feature = torch.zeros((self.num_clips, self.num_sensor, self.graph_ch), dtype=torch.float)
                edge = torch.zeros((self.num_clips, self.num_sensor, self.num_sensor), dtype=torch.float)

                edge = edge.to(self.device)
                node_feature = node_feature.to(self.device)

                loss = {}

                # =================================================================================== #
                #                             2. Train the Model                                      #
                # =================================================================================== #
                recon, forecast, node_recon = self.G(node_feature, edge)

                recon_error = self.loss_function(recon, node_feature[-1])
                forecast_error = self.loss_function(forecast, node_feature[-1])

                patient_score = (self.rx_w * recon_error
                                    + (1 - self.rx_w) * forecast_error)
                sensor_score = self.loss_function(node_recon, node_feature[-1])

                self.reset_grad()
                del recon, forecast, node_recon, recon_error, forecast_error, patient_score, sensor_score
                del node_feature, edge
                torch.cuda.empty_cache()