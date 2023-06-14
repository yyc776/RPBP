import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_sparse
from torch_scatter import scatter_max, scatter_add


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf


def build_mlp(in_dim, h_dim, out_dim=None, dropout_p=0.2):
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """
    if isinstance(h_dim, int):
        h_dim = [h_dim]

    sizes = [in_dim] + h_dim
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        layers.append(nn.Linear(prev_size, next_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        layers.append(nn.Linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)


class MPNLayer(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings."""

    def __init__(self,
                 # rnn_type: str,
                 node_fdim: int,
                 edge_fdim: int,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        rnn_type: str,
            Type of RNN used (gru/lstm)
        input_size: int,
            Input size
        node_fdim: int,
            Number of node features
        hsize: int,
            Hidden state size
        depth: int,
            Number of timesteps in the RNN
        """
        super(MPNLayer, self).__init__(**kwargs)
        self.hsize = hsize
        self.edge_fdim = edge_fdim
        # self.rnn_type = rnn_type
        self.depth = depth
        self.node_fdim = node_fdim
        self.dropout_p = dropout_p
        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNLayer."""
        # self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.hsize - 65, self.hsize), nn.ReLU())
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.hsize, self.hsize), nn.ReLU())
        # if self.rnn_type == 'gru':
        self.rnn = GRU(input_size=self.node_fdim + self.edge_fdim,
                       hsize=self.hsize,
                       depth=self.depth,
                       dropout_p=self.dropout_p)

        # elif self.rnn_type == 'lstm':
        #     self.rnn = LSTM(input_size=self.node_fdim + self.edge_fdim,
        #                    hsize=self.hsize,
        #                    depth=self.depth,
        #                    dropout_p=self.dropout_p)
        # else:
        #     raise ValueError('unsupported rnn cell type ' + self.rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph, mask):
        """Forward pass of the MPNLayer.

        Parameters
        ----------
        fnode: torch.Tensor,
            Node feature tensor
        fmess: torch.Tensor,
            Message features
        agraph: torch.Tensor,
            Neighborhood of an atom
        bgraph: torch.Tensor,
            Neighborhood of a bond, except the directed bond from the destination
            node to the source node
        mask: torch.Tensor,
            Masks on nodes
        """
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0  # first node is padding

        return node_hiddens * mask, h


class GRU(nn.Module):
    """GRU Message Passing layer."""

    def __init__(self,
                 input_size: int,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        input_size: int,
            Size of the input
        hsize: int,
            Hidden state size
        depth: int,
            Number of time steps of message passing
        device: str, default cpu
            Device used for training
        """
        super(GRU, self).__init__(**kwargs)
        self.hsize = hsize
        self.input_size = input_size
        self.depth = depth
        self.dropout_p = dropout_p
        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.hsize, self.hsize)
        self.W_r = nn.Linear(self.input_size, self.hsize, bias=False)
        self.U_r = nn.Linear(self.hsize, self.hsize)
        self.W_h = nn.Linear(self.input_size + self.hsize, self.hsize)

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)

    def get_init_state(self, fmess: torch.Tensor, init_state: torch.Tensor = None) -> torch.Tensor:
        """Get the initial hidden state of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        init_state: torch.Tensor, default None
            Custom initial state supplied.
        """
        h = torch.zeros(len(fmess), self.hsize, device=fmess.device)
        return h if init_state is None else torch.cat((h, init_state), dim=0)

    def get_hidden_state(self, h: torch.Tensor) -> torch.Tensor:
        """Gets the hidden state.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state of the GRU
        """
        return h

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        """
        sum_h = h_nei.sum(dim=1)
        z_input = torch.cat([x, sum_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.hsize)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * sum_h + z * pre_h
        return new_h

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size(0), self.hsize, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0  # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
            h = self.dropouts[i](h)
        return h

    def sparse_forward(self, h: torch.Tensor, fmess: torch.Tensor,
                       submess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Unknown use.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state tensor
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        submess: torch.Tensor,
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h


class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs):
        """
        Parameters
        ----------
        input_size: int,
            Size of the input
        hsize: int,
            Hidden state size
        depth: int,
            Number of time steps of message passing
        device: str, default cpu
            Device used for training
        """
        super(LSTM, self).__init__(**kwargs)
        self.hsize = hsize
        self.input_size = input_size
        self.depth = depth
        self.dropout_p = dropout_p
        self._build_layer_components()

    def _build_layer_components(self):
        """Build layer components."""
        self.W_i = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W_o = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W_f = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Tanh())

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)

    def get_init_state(self, fmess, init_state=None):
        """Get the initial hidden state of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        init_state: torch.Tensor, default None
            Custom initial state supplied.
        """
        h = torch.zeros(len(fmess), self.hsize, device=fmess.device)
        c = torch.zeros(len(fmess), self.hsize, device=fmess.device)
        if init_state is not None:
            h = torch.cat((h, init_state), dim=0)
            c = torch.cat((c, torch.zeros_like(init_state)), dim=0)
        return h, c

    def get_hidden_state(self, h):
        """Gets the hidden state.

        Parameters
        ----------
        h: Tuple[torch.Tensor, torch.Tensor],
            Hidden state tuple of the LSTM
        """
        return h[0]

    def LSTM(self, x: torch.Tensor, h_nei: torch.Tensor, c_nei: torch.Tensor) -> torch.Tensor:
        """Implements the LSTM gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        c_nei: torch.Tensor,
            Memory state of the neighbors
        """
        h_sum_nei = h_nei.sum(dim=1)
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i(torch.cat([x, h_sum_nei], dim=-1))
        o = self.W_o(torch.cat([x, h_sum_nei], dim=-1))
        f = self.W_f(torch.cat([x_expand, h_nei], dim=-1))
        u = self.W(torch.cat([x, h_sum_nei], dim=-1))
        c = i * u + (f * c_nei).sum(dim=1)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, fmess, bgraph):
        """Forward pass of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size(0), self.hsize, device=fmess.device)
        c = torch.zeros(fmess.size(0), self.hsize, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0  # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            h, c = self.LSTM(fmess, h_nei, c_nei)
            h = h * mask
            c = c * mask
            h = self.dropouts[i](h)
            c = self.dropouts[i](c)
        return h, c

    def sparse_forward(self, h: torch.Tensor, fmess: torch.Tensor,
                       submess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Unknown use.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state tensor
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        submess: torch.Tensor,
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h, c = h
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        c = c * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            sub_h, sub_c = self.LSTM(fmess, h_nei, c_nei)
            h = index_scatter(sub_h, h, submess)
            c = index_scatter(sub_c, c, submess)
        return h, c


class GraphFeatEncoder(nn.Module):
    """
    GraphFeatEncoder encodes molecules by using features of atoms and bonds,
    instead of a vocabulary, which is used for generation tasks.
    """

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 # rnn_type: str,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        node_fdim: int,
            Number of atom features
        edge_fdim: int,
            Number of bond features
        rnn_type: str,
            Type of RNN used for encoding
        hsize: int,
            Hidden state size
        depth: int,
            Number of timesteps in the RNN
        """
        super(GraphFeatEncoder, self).__init__(**kwargs)
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        # self.rnn_type = rnn_type
        self.atom_size = node_fdim
        self.hsize = hsize
        self.depth = depth
        self.dropout_p = dropout_p

        self._build_layers()

    def _build_layers(self):
        """Build layers associated with the GraphFeatEncoder."""
        self.encoder = MPNLayer(
            # rnn_type=self.rnn_type,
            edge_fdim=self.edge_fdim,
            node_fdim=self.node_fdim,
            hsize=self.hsize, depth=self.depth,
            dropout_p=self.dropout_p)

    def embed_graph(self, graph_tensors):
        """Replaces input graph tensors with corresponding feature vectors.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details,
            atom graph and bond graph for encoding neighborhood connectivity.
        """
        fnode, fmess, agraph, bgraph = graph_tensors
        hnode = fnode.clone()
        fmess1 = hnode.index_select(index=fmess[:, 0].long(), dim=0)
        fmess2 = fmess[:, 2:].clone()
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        return hnode, hmess, agraph, bgraph

    def forward(self, graph_tensors, atom_scopes):
        """
        Forward pass of the graph encoder. First the feature vectors are extracted,
        and then encoded.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details,
            atom graph and bond graph for encoding neighborhood connectivity.
        scopes: Tuple[List]
            Scopes is composed of atom and bond scopes, which keep track of
            atom and bond indices for each molecule in the 2D feature list
        """
        tensors = self.embed_graph(graph_tensors)
        hatom, _ = self.encoder(*tensors, mask=None)

        if isinstance(atom_scopes[0], list):
            hmol = [torch.stack([hatom[st: st + le].sum(dim=0) for (st, le) in scope])
                    for scope in atom_scopes]
        else:
            hmol = torch.stack([hatom[st: st + le].sum(dim=0) for st, le in atom_scopes])
        return hatom, hmol


class MLP(nn.Module):
    def __init__(self, nhid, nclass):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(nhid, nclass)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x):
        x = self.mlp(x)

        return x


class GraphConv(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
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
        # for 3_D batch, need a loop!!!

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
            in_features, out_perhead, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func.
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
            self.in_features, self.heads, self.out_perhead)


class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = Parameter(torch.zeros(size=(1, 2 * out_features)))
        # init
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze())  # E
        n = len(input)
        alpha = self.softmax(alpha, edge[0], n)
        output = torch_sparse.spmm(edge, self.dropout(alpha), n, n, h)  # h_prime: N x out
        # output = torch_sparse.spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output

    def softmax(self, src, index, num_nodes=None):
        """
        sparse softmax
        """
        num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
        out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
        out = out.exp()
        out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
        return out


class GNN_Classifier(nn.Module):
    def __init__(self, layer, nhid, nclass, dropout, nhead=1):
        super(GNN_Classifier, self).__init__()
        if layer == 'gcn':
            self.conv = GraphConv(nhid, nhid)
            self.activation = nn.ReLU()
        elif layer == 'gat':
            self.conv = GraphAttConv(nhid, nhid, nhead, dropout)
            self.activation = nn.ELU(True)

        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj, atom_scopes):
        if x.shape[0] - adj.shape[0] == 1:
            x = self.activation(self.conv(x[1:], adj))
            x = self.dropout(x)
            x = torch.stack([x[st - 1: st - 1 + le].sum(dim=0) for st, le in atom_scopes])
        else:
            x = self.activation(self.conv(x, adj))
            x = self.dropout(x)
            x = torch.stack([x[st: st + le].sum(dim=0) for st, le in atom_scopes])
        x = self.mlp(x)
        return x


class Decoder(nn.Module):
    """
    Edge Reconstruction adopted in GraphSMOTE (https://arxiv.org/abs/2103.08826)
    """

    def __init__(self, nhid, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nhid, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))

        return adj_out
