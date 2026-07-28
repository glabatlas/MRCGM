import torch.nn as nn
from JCMA.dense_coattn import DenseCoAttn


class NormalSubLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3, dropout):
        super(NormalSubLayer, self).__init__()
        self.dense_coattn = DenseCoAttn(dim1, dim2, dim3, dropout)


        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16165,16165),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(16165,16165),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(16165,16165),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])

    def forward(self, data1, data2, data3):
        weighted1, weighted2, weighted3 = self.dense_coattn(data1, data2, data3)


        data1 = data1.transpose(-2, -1) + self.linears[0](weighted1)
        data2 = data2.transpose(-2, -1) + self.linears[1](weighted2)
        data3 = data3.transpose(-2, -1) + self.linears[2](weighted3)

        data1=data1.transpose(-2, -1)
        data2=data2.transpose(-2, -1)
        data3=data3.transpose(-2, -1)

        return data1, data2, data3,(weighted1, weighted2, weighted3)

class DCNLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3, num_seq, dropout):
        super(DCNLayer, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer(dim1, dim2, dim3, dropout) for _ in range(num_seq)])

    def forward(self, data1, data2, data3):
        all_weighted = []

        for dense_coattn in self.dcn_layers:
            data1, data2, data3, (w1, w2, w3)= dense_coattn(data1, data2, data3)
            all_weighted.append((w1, w2, w3))
        return data1, data2, data3, all_weighted