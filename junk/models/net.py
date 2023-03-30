import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torch.nn.parameter import Parameter
import math


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes: typing.Iterable[int], out_dim, activation_function=nn.Sigmoid(),
                 activation_out=None):
        super(MLP, self).__init__()

        i_h_sizes = [input_dim] + hidden_sizes  # add input dim to the iterable
        self.mlp = nn.Sequential()
        for idx in range(len(i_h_sizes) - 1):
            self.mlp.add_module("layer_{}".format(idx),
                                nn.Linear(in_features=i_h_sizes[idx], out_features=i_h_sizes[idx + 1]))
            self.mlp.add_module("act_{}".format(idx), activation_function)
        self.mlp.add_module("out_layer", nn.Linear(i_h_sizes[-1], out_dim))
        if activation_out is not None:
            self.mlp.add_module("out_layer_activation", activation_out)

    def init(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


# code from Pedro H. Avelar

class StateTransition(nn.Module):

    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        d_i = node_state_dim + 2 * node_label_dim  # arc state computation f(l_v, l_n, x_n)
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)  # if already a list, no change
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)  # state transition function, non-linearity also in output

    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,
    ):
        src_label = node_labels[edges[:, 0]]
        tgt_label = node_labels[edges[:, 1]]
        tgt_state = node_states[edges[:, 1]]
        edge_states = self.mlp(
            torch.cat(
                [src_label, tgt_label, tgt_state],
                -1
            )
        )

        new_state = torch.matmul(agg_matrix, edge_states)
        return new_state



class GINTransition(nn.Module):

    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        d_i = node_state_dim + node_label_dim
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)  # state transition function, non-linearity also in output

    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,

    ):
        state_and_label = torch.cat(
            [node_states, node_labels],
            -1
        )
        aggregated_neighbourhood = torch.matmul(agg_matrix, state_and_label[edges[:, 1]])
        node_plus_neighbourhood = state_and_label + aggregated_neighbourhood
        new_state = self.mlp(node_plus_neighbourhood)
        return new_state

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GINPreTransition(nn.Module):

    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        d_i = node_state_dim +  node_label_dim
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)

    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,
    ):
        intermediate_states = self.mlp(
            torch.cat(
                [node_states, node_labels],
                -1
            )
        )
        new_state = (
                torch.matmul(agg_matrix, intermediate_states[edges[:, 1]])
                + torch.matmul(agg_matrix, intermediate_states[edges[:, 0]])
        )
        return new_state
