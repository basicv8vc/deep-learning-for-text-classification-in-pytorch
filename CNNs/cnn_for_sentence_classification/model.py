import torch
import torch.functional as F
from torch import nn


class Model(nn.Module):
    r"""
    Implementing "Convolutional Neural Networks for Sentence Classification"

    """
    def __init__(self, args):
        super(Model, self).__init__()

        self.l2_norm = args.l2_norm
        self.Ci = 2 if args.multichannel else 1
        self.Co = args.feature_maps

        self.embeddings = nn.ModuleList([nn.embedding(args.num_embeddings, args.embedding_dim)
                                         for i in range(self.Ci)])
        if args.multichannel:
            self.embeddings[-1].weight.requires_grad = False
        self.conv_2ds = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (k, args.embedding_dim))
                                       for k in args.filter_windows])
        self.conv_2ds_mc = nn.ModuleList([nn.Conv2d(self.Ci*2, self.Co, (k, args.embedding_dim))
                                         for k in args.filter_windows])
        self.maxpool1d = nn.ModuleList([nn.MaxPool1d(args.embedding_dim-k+1)
                                        for k in args.filter_windows])

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(self.Co * len(args.filter_windows), args.classes)

        self.forward = self.forward_multi if args.multichannel else self.forward_static

    def forward_static(self, x):
        embeds = self.embeddings[0](x)  # (N, 1, seq_len, embed_dim)
        fmaps = [torch.squeeze(F.relu(layer(embeds)), 3) for layer in self.conv_2ds] # [(N, Co, seq_len-k+1),...]
        feas = torch.cat([torch.squeeze(self.maxpool1d[i](fmaps[i]))
                          for i in range(self.conv_2ds)], 1) # (N, Co*len(args.filter_windows)), defalut (N, Co*3)
        feas = self.dropout(feas)
        outputs = self.linear(feas)

        return outputs

    def forward_multi(self, x):
        embeds = [layer(x) for layer in self.embeddings]
        inputs = torch.cat(embeds, 1)   # (N, 2, seq_len, embed_dim)
        fmaps = [torch.squeeze(F.relu(layer(embeds)), 3) for layer in self.conv_2ds_mc]  # [(N, Co, seq_len-k+1),...]
        feas = torch.cat([torch.squeeze(self.maxpool1d[i](fmaps[i]))
                          for i in range(self.conv_2ds)], 1)  # (N, Co*len(args.filter_windows)), defalut (N, Co*3)
        feas = self.dropout(feas)
        outputs = self.linear(feas)

        return outputs

    def initialize_embedding(self, embedding_matrix):
        for i, layer in enumerate(self.embeddings):
            self.embeddings[i].weight.data.copy_(torch.from_numpy(embedding_matrix))
        if self.Ci == 2:
            self.embeddings[-1].weight.requires_grad = False

