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
# from torch.autograd.gradcheck import zero_gradients
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GAD(object) :
    def __init__(self, args):
        self.model_name = 'Graph_Conv_AE'
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.dataset_name = args.dataset

        self.train_len = args.dataset_setting[self.dataset_name][0]
        self.test_len = args.dataset_setting[self.dataset_name][1]
        self.train_size = args.dataset_setting[self.dataset_name][2]
        self.test_size = args.dataset_setting[self.dataset_name][3]
        self.new_start = args.new_start

        self.epoch = args.epoch
        # self.iteration = args.iteration##
        self.resume_iters = args.resume_iters
        self.denoising = args.dropout

        self.loss_function = eval(args.loss_function)
        self.ax_w = args.nx_w
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch
        self.init_lr = args.lr

        self.print_net = args.print_net

        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.use_tensorboard = args.use_tensorboard

        self.batch_size = args.batch_size
        self.num_clips = args.num_clips
        self.graph_ch = args.dataset_setting[self.dataset_name][5]
        self.conv_ch = args.dataset_setting[self.dataset_name][6] if args.conv_ch == 0 else args.conv_ch

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # build graph
        print(" [*] Buliding model!")
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        print("##### Information #####")
        print("# loss function: ", args.loss_function)
        print("# dataset : ", self.dataset_name)
        #print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        #print("# iteration per epoch : ", self.iteration)

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

        print('Saved model {} checkpoints into {}...'.format(counter+1, self.model_save_dir))

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
        self.G = GAE(self.graph_ch,self.conv_ch)

        if self.print_net:
            self.print_network(self.G, 'G')

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.init_lr)

        self.G.to(self.device)

    def egde_weight(self,egde,min=-1,max=1):
        return  egde * (max-min) + min

    def thresold(self):
        self.Th_error = np.array([0., 0.])
        results_list = []
        for idx in range(self.iteration):
            node_path = self.dataset_name + '/node/node' + str(idx + 1) + '.npy'
            edge_path = self.dataset_name + '/graph/graph' + str(idx + 1) + '.npy'
            dict_path = self.dataset_name + '/dict/node_dict' + str(idx + 1) + '.npy'  if self.dataset_name != 'DBLP5' else None
            node_feature, edge, _, _ = load_graph(node_path, edge_path, dict_path)

            node_feature = node_feature.float()
            edge = edge.float()


            help = torch.eye(edge.shape[1], dtype=torch.float)

            edge = torch.abs(torch.mul(1. - help, edge))

            edge=edge.to(self.device)
            node_feature=node_feature.to(self.device)

            recon_a, recon_x, _ = self.G(node_feature, edge)
            #recon_a = self.egde_weight(recon_a)

            recon_a_error = self.loss_function(recon_a, edge, graph=False)
            recon_x_error = self.loss_function(recon_x, node_feature, graph=False)

            results_list.append(torch.flatten(self.ax_w * recon_a_error
                                 + (1 - self.ax_w) * recon_x_error).cpu().detach().numpy())

        results_list = np.array(results_list)

        self.Th_error[0] = np.mean(results_list)
        self.Th_error[1] = np.std(results_list)
        self.model_save_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        np.save(self.model_save_dir + '_threshold.npy', self.Th_error)
        print(self.Th_error)

    def train(self):
        start_iters = self.resume_iters if not self.new_start else 0
        self.restore_model(self.resume_iters)

        self.iteration = self.train_len
        self.graph_size = self.train_size

        start_epoch = (int)(start_iters / self.iteration)
        start_batch_id = start_iters - start_epoch * self.iteration

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr

        self.set_requires_grad([self.G], True)

        self.G.train()

        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag and epoch > self.decay_epoch:
                lr = self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay
                self.update_lr(lr)

            for idx in range(start_batch_id, self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                node_path = self.dataset_name + '/node/node' + str(idx+1) + '.npy'
                edge_path = self.dataset_name + '/graph/graph' + str(idx+ 1) + '.npy'
                node_feature, edge, _, _ = load_graph(node_path, edge_path)

                node_feature = node_feature.float()
                edge = edge.float()

                help = torch.eye(edge.shape[1], dtype=torch.float)

                node_exist = torch.sum(torch.mul(help, edge), dim=-1)  # whether or not the node exists
                edge = torch.abs(torch.mul(1. - help, edge)).to(self.device)

                node_feature = node_feature.to(self.device)

                loss = {}

                # =================================================================================== #
                #                             2. Train the Auto-encoder                              #
                # =================================================================================== #
                recon_a, recon_x, _ = self.G(node_feature, edge)

                self.recon_a_error = self.loss_function(recon_a, edge)
                self.recon_x_error = self.loss_function(recon_x, node_feature)

                self.Reconstruction_error = (self.ax_w * self.recon_a_error
                                       + (1 - self.ax_w) * self.recon_x_error)

                # Logging.
                loss['Edge_reconstruction_error'] = self.recon_a_error.item()
                loss['feature_reconstruction_error'] = self.recon_x_error.item()
                # loss['G/loss_cycle'] = self.cycle_loss.item()
                loss['Reconstruction_error'] = self.Reconstruction_error.item()


                del recon_a
                del recon_x
                torch.cuda.empty_cache()

                self.reset_grad()
                self.Reconstruction_error.backward()
                self.g_optimizer.step()


                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #
                start_iters += 1

                # Print out training information.
                if idx  % self.print_freq == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]".format(et, epoch+1, self.epoch, idx + 1, self.iteration)
                    for tag, value in loss.items():
                        if 'error' in tag:# != 'G/lable' and tag !='O/lable':
                            log += ", {}: {:.4f}".format(tag, value)
                            if self.use_tensorboard:
                                self.logger.scalar_summary(tag, value, start_iters)
                    print(log)

                torch.cuda.empty_cache()

                # Save model checkpoints.
                if (idx + 1) % self.save_freq == 0:
                    self.save(self.checkpoint_dir, start_iters)
                    torch.cuda.empty_cache()

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, start_iters)
            torch.cuda.empty_cache()

        #caculat thresold
        self.thresold()


    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def test(self):
        self.restore_model(self.resume_iters)

        self.G.eval()

        self.iteration = self.test_len
        self.graph_size = self.test_size
        self.model_save_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.Th_error = np.load(self.model_save_dir + '_threshold.npy')

        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.

        with torch.no_grad():
            for idx in range(self.iteration):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                node_path = self.dataset_name + '/node/testnode' + str(idx + 1) + '.npy'
                edge_path = self.dataset_name + '/graph/testgraph' + str(idx + 1) + '.npy'
                ab_path = self.dataset_name + '/abnormal/abnormal' + str(idx + 1) + '.npy'
                node_feature, edge, _, abnormal = load_graph(node_path, edge_path, abnormal_path=ab_path)

                help = torch.eye(edge.shape[1], dtype=torch.float)

                node_exist = torch.sum(torch.mul(help, edge), dim=-1)  # whether or not the node exists
                edge = torch.abs(torch.mul(1. - help, edge).to(self.device))
                node_feature = node_feature.to(self.device)

                # =================================================================================== #
                #                             2. Train the Auto-encoder                              #
                # =================================================================================== #
                recon_a, recon_x, node_embedding = self.G(node_feature, edge)

                a_score = self.loss_function(recon_a, edge, graph=False)
                x_score = self.loss_function(recon_x, node_feature, graph=False)

                Anomaly_score = (self.ax_w * a_score
                                 + (1 - self.ax_w) * x_score)

                record1 = (((Anomaly_score - self.Th_error[0]) / self.Th_error[1]) > 1).float().cpu()

                #record1 = (Anomaly_score > self.Th_error[0]).float().cpu()

                for n in range(record1.shape[0]):
                    if record1[n] == abnormal[n]:
                        if record1[n] == 0:
                            tn += 1.
                        else:
                            tp += 1.
                    else:
                        if record1[n] == 0:
                            fn += 1.
                        else:
                            fp += 1.

                torch.cuda.empty_cache()


        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp/(tp+ fn)
        prec = tp/(tp+ fp)
        f1 = 2*(recall*prec)/(recall+prec)
        tnr = tn / (tn + fp)
        print(acc)
        print(recall)
        print(prec)
        print(f1)
        print(tnr)