import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_adj
from torch.nn import Dropout, MaxPool1d
import config_load

configs = config_load.get()
HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2

class MRCGM(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, residual, devices_available):
        super(MRCGM, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate
        self.convs = t.nn.ModuleList()
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate,
                                          edge_dim=edge_dim, concat=False, beta=True).to(devices_available))
        self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                          dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True).to(devices_available))
        self.ln1 = LayerNorm(in_channels=mid_channels).to(devices_available)
        self.ln2 = LayerNorm(in_channels=hidden_channels * heads).to(devices_available)
        self.pool = MaxPool1d(2, 2)
        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList()
        self.lins.append(Linear(int(hidden_channels*heads/2), HIDDEN_DIM, weight_initializer="kaiming_uniform").to(devices_available))
        self.lins.append(Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform").to(devices_available))

    def forward(self, data):
        data = data[0].to(self.devices_available)
        x = data.x

        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)

        res = x

        x = self.convs[0](x, edge_index, edge_attr)

        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x

        x = self.ln1(x)
       
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        x = self.convs[1](x.to(self.devices_available), edge_index.to(
            self.devices_available), edge_attr.to(self.devices_available))

        x = self.ln2(x)
      
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = t.unsqueeze(x, 1)
        x = self.pool(x)

        x = t.squeeze(x)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        return t.sigmoid(x)


