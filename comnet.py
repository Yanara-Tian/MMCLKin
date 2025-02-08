from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits
from torch_scatter import scatter, scatter_min
import sympy as sym
from torch.nn import Embedding
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential
import math
from math import sqrt
from features import angle_emb, torsion_emb

def swish(x):
    return x * torch.sigmoid(x)

class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock1(torch.nn.Module):
    def __init__(
            self,
            node_dim,
            edge_dim,
            hidden_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            device,
            act=swish
    ):
        super(SimpleInteractionBlock1, self).__init__()
        
        self.act = act
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)
    
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, hidden_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, hidden_channels, hidden_channels)

        # Dense transformations of input messages.
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()
    
    def forward(self, x, feature1, feature2, edge_index):
        x = self.act(self.lin(x))  # torch.Size([35, 64])
        
        #local conv
        feature1 = self.lin_feature1(feature1)  #torch.Size([1650, 12])
        h1 = self.conv1(x, edge_index, feature1)  #161,512  2,326  326,512  30,256,2,68,68,256
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        #global conv
        feature2 = self.lin_feature2(feature2)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        # concat + down-project
        h = self.lin_cat(torch.cat([h1, h2], 1))
        # sum
        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h)
        h = self.final(h)
        return x


from torch_geometric.nn import NNConv, GATConv
class EComENet(nn.Module):
    def __init__(
            self,
            device,
            cutoff=8.0,
            num_layers=4,
            node_dim=75,
            edge_dim=8,
            hidden_channels=64,
            out_channels=64,
            num_radial=3,
            num_spherical=2,
            num_output_layers=3,
            mol_edge_in_dim=8
    ):
        super(EComENet, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act
        self.device=device
        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.edge_lin = Linear(edge_dim, hidden_channels)
        self.emb = EmbeddingBlock(hidden_channels, act)
        self.line = Linear(node_dim, hidden_channels)
        nn = Sequential(Linear(mol_edge_in_dim, node_dim* node_dim))
        self.conv = NNConv(node_dim, node_dim, nn, aggr='mean')
        self.interaction_blocks = torch.nn.ModuleList(
            [SimpleInteractionBlock1(
                    node_dim,
                    edge_dim,
                    hidden_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    out_channels,
                    device,
                    act
                )
                for _ in range(num_layers)
            ]
        )

        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_lin.reset_parameters()
        self.conv.reset_parameters()
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def _forward(self, dist, theta, phi, tau, x, edge_index, edge_feats):
        x = F.relu(self.conv(x, edge_index, edge_feats))
        x = self.line(x)
        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index)
        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin_out(x) 
        
        return x

    def forward(self, dist, theta, phi, tau, atom_feats, edge_index, edge_feats):
        return self._forward(dist, theta, phi, tau, atom_feats, edge_index, edge_feats)
        
