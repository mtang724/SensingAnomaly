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

from torch.nn.utils import clip_grad_norm_ as clip_grad_norm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DGAD_rnn(object):
    def __init__(self, args):
        self.model_name = args.model
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
        self.edge_corr = args.edge_corr
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

        self.device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available() else

        if 'har' in self.dataset_name:
            if 'clean' in self.dataset_name:
                train_list_path = self.dataset_name + '/train_clean_list.npy'
                train_label_path = self.dataset_name + '/train_clean_label.npy'
            else:
                train_list_path = self.dataset_name + '/train_list.npy'
                train_label_path = self.dataset_name + '/train_label.npy'
            test_list_path = self.dataset_name + '/test_list.npy'
            test_label_path = self.dataset_name + '/test_label.npy'
            subject_train_path = self.dataset_name + '/subject_train.npy'
            subject_test_path = self.dataset_name + '/subject_test.npy'
            abnormal_list_path = self.dataset_name + '/ab.npy'

            dataset = load_har(train_list_path, train_label_path, test_list_path, test_label_path, subject_train_path, subject_test_path, abnormal_list_path)

            self.train_list = dataset[0]
            self.train_label = dataset[1]
            self.test_list = dataset[2]
            self.test_label = dataset[3]
            self.subject_train = dataset[4]
            self.subject_test  = dataset[5]
            self.abnormal_list = dataset[6]

            del dataset

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
        self.G = Detector_rnn(self.graph_ch, self.conv_ch, self.num_sensor, self.embedding_dim, self.head, self.dropout,
                          self.original_dim)

        if self.print_net:
            self.print_network(self.G, 'G')

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.init_lr)

        self.G = self.G.to(self.device)

    def load_data_train(self, idx):
        assert 'har' in self.dataset_name

        start = int(self.subject_train[idx, 1])
        end = int(self.subject_train[idx, 2])

        node_feature = self.train_list[start:end+1]

        if self.edge_corr:
            edge_list_path = self.dataset_name + '/edge.npy'
            edge = np.load(edge_list_path, allow_pickle=True)
            edge = torch.tensor(edge, dtype=torch.float)
            edge = edge.expand([node_feature.shape[0], -1, -1])
            edge = edge.to(self.device)
        else:
            edge = None

        node_feature = node_feature.to(self.device)

        sensor = [i for i in range(self.num_sensor)]
        sensor = torch.tensor(sensor, dtype=torch.int64)
        sensor = sensor.to(self.device)

        return node_feature, edge, sensor, idx

    def load_data_test(self, idx):
        assert 'har' in self.dataset_name

        start = int(self.subject_test[idx, 1])
        end = int(self.subject_test[idx, 2])

        node_feature = self.test_list[start:end+1]
        abnormal = self.abnormal_list[start:end+1]
        graph_abnormal = (self.test_label[start:end+1] == 3)

        if self.edge_corr:
            edge_list_path = self.dataset_name + '/edge.npy'
            edge = np.load(edge_list_path, allow_pickle=True)
            edge = torch.tensor(edge, dtype=torch.float)
            edge = edge.expand([node_feature.shape[0], -1, -1])
            edge = edge.to(self.device)
        else:
            edge = None

        node_feature = node_feature.to(self.device)

        sensor = [i for i in range(self.num_sensor)]
        sensor = torch.tensor(sensor, dtype=torch.int64).to(self.device)

        return node_feature, edge, sensor, abnormal, graph_abnormal, idx

    def train(self):
        start_iters = self.resume_iters if not self.new_start else 0
        self.restore_model(self.resume_iters)

        if self.model_name == 'RNN':
            self.iteration = len(self.subject_train)
        else:
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

            self.subject = 0
            idx = start_batch_id
            while idx < self.iteration:
                # =================================================================================== #
                #                             1. Preprocess input data(Unfinisher)                    #
                # =================================================================================== #
                node_feature, edge, sensor, idx = self.load_data_train(idx)

                loss = {}

                # x = next(self.G.parameters()).is_cuda

                # =================================================================================== #
                #                             2. Train the Model                                      #
                # =================================================================================== #
                recon, forecast, node_recon = self.G(node_feature, edge, sensor)

                self.recon_error = self.loss_function(recon, node_feature[1:], device=self.device)
                self.forecast_error = self.loss_function(forecast, node_feature[1:], device=self.device)

                self.Graph_error = (self.rx_w * self.recon_error
                                    + (1 - self.rx_w) * self.forecast_error)
                self.Node_error = self.loss_function(node_recon, node_feature[1:], device=self.device)

                self.Error = (self.nx_w * self.Node_error
                              + (1 - self.nx_w) * self.Graph_error)

                if torch.isnan(self.Error):
                    np.save('x.npy', node_feature.detach().cpu().numpy())
                    np.save('n.npy', node_recon.detach().cpu().numpy())

                # Logging.
                loss['reconstruction_error'] = self.recon_error.item()
                loss['Forecast_error'] = self.forecast_error.item()
                loss['Graph_error'] = self.Graph_error.item()
                loss['Node_error'] = self.Node_error.item()
                loss['Whole_error'] = self.Error.item()

                del recon, forecast, node_recon, node_feature, edge, sensor
                torch.cuda.empty_cache()

                self.reset_grad()
                self.Error.backward()
                clip_grad_norm(self.G.parameters(), 5)
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

                idx += 1

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

        if self.model_name == 'RNN':
            self.iteration = len(self.subject_test)
        else:
            self.iteration = self.test_len - self.num_clips + 1

        self.model_save_dir = os.path.join(self.checkpoint_dir, self.model_dir)

        node_predict = []
        graph_predict = []

        node_abnormal = []
        graph_ab_list = []

        with torch.no_grad():
            idx = 0
            self.subject = 0
            while idx < self.iteration:
                # =================================================================================== #
                #                             1. Preprocess input data(Unfinished)                    #
                # =================================================================================== #
                node_feature, edge, sensor, abnormal, graph_abnormal, idx = self.load_data_test(idx)

                node_abnormal.append(abnormal[1:])
                graph_ab_list.append(graph_abnormal[1:].view(-1))

                # =================================================================================== #
                #                             2. Train the Model                                      #
                # =================================================================================== #
                recon, forecast, node_recon = self.G(node_feature, edge, sensor)

                recon_error = self.loss_function(recon, node_feature[1:], graph=False, device=self.device)
                forecast_error = self.loss_function(forecast, node_feature[1:], graph=False, device=self.device)

                patient_score = (self.rx_w * recon_error
                                    + (1 - self.rx_w) * forecast_error)
                sensor_score = self.loss_function(node_recon, node_feature[1:], graph=False, device=self.device)
                error = self.nx_w * patient_score + (1 - self.nx_w) * sensor_score

                if graph_abnormal is None:
                    node_predict.append(error.cpu())
                else:
                    node_predict.append(sensor_score.cpu())
                    graph_predict.append(torch.mean(patient_score,dim=-1).cpu())

                del recon, forecast, node_recon, recon_error, forecast_error, patient_score, sensor_score, error
                del node_feature, edge
                torch.cuda.empty_cache()

                idx += 1

        print("Finish NN Part!")

        node_abnormal = torch.cat(node_abnormal)
        node_predict = torch.cat(node_predict)
        predict_result(node_predict, node_abnormal, "sensor")
        if graph_abnormal is not None:
            graph_ab_list = torch.cat(graph_ab_list)
            graph_predict = torch.cat(graph_predict)
            predict_result(graph_predict, graph_ab_list, 'graph')