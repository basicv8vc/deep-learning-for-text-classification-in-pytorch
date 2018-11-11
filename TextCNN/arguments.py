import argparse
import os
import torch

class Arguments():
    """Base class of model's arguments

    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--feature_maps', type=int, default=100, help='# of feature maps')
        self.parser.add_argument('--filter_windows', type=list, default=[3, 4, 5], help='filter windows used in convolution')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout used in fully connected layer')
        self.parser.add_argument('--l2_norm', type=int, default=3, help='l2-norm constrait used in fully connected layer')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--multichannel', type=bool, default=False, help='choose which model to use. CNN-multichannel, CNN-non-static')

        self.parser.add_argument('--num_embeddings', type=int, default=0, help='vocabulary size')
        self.parser.add_argument('--embedding_dim', type=int, default=50, help='dimension of word embedding')
        self.parser.add_argument('--classes', type=int, default=-1, help='# of classes')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt

