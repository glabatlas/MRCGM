import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
from JCMA.av_crossattn import DCNLayer

class JCMA(nn.Module):
    def __init__(self):
        super(JCMA, self).__init__()

        self.modality = ['omics', 'hic', 'pathway']
        self.rows=None


        self.feature_dim_omics = 10
        self.feature_dim_hic = 5
        self.feature_dim_pathway = 5


        self.bn_omics = BatchNorm1d(self.feature_dim_omics)
        self.bn_hic = BatchNorm1d(self.feature_dim_hic)
        self.bn_pathway = BatchNorm1d(self.feature_dim_pathway)


        self.coattn = DCNLayer(self.feature_dim_omics, self.feature_dim_hic, self.feature_dim_pathway, num_seq=1, dropout=0.6)
        self.mlp = nn.Sequential(
            nn.Linear(20, 20),
            nn.LayerNorm(20),
            nn.LeakyReLU()
        )

    def forward(self, X):

        device = next(self.parameters()).device


        omics = X[:, :10].to(device)
        hic = X[:, 10:15].to(device)
        pathway = X[:, 15:20].to(device)


        if omics.dtype != torch.float32:
            omics = omics.to(torch.float32)
        omics = self.bn_omics(omics)

        if hic.dtype != torch.float32:
            hic = hic.to(torch.float32)
        hic = self.bn_hic(hic)

        if pathway.dtype != torch.float32:
            pathway = pathway.to(torch.float32)
        pathway = self.bn_pathway(pathway)


        omics, hic, pathway, all_weighted = self.coattn(omics, hic, pathway)
        weighted1, weighted2, weighted3 = all_weighted[0]
        weighted1 = weighted1.transpose(0, 1)  # [16165, 10]
        weighted2 = weighted2.transpose(0, 1)
        weighted3 = weighted3.transpose(0, 1)


        c = torch.cat((omics, hic, pathway), dim=-1)
        c = self.mlp(c)
        return c
