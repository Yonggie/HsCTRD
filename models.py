from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.drop import *
import torch.nn.functional as F
import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import GCNConv,GATConv
from utils.initialization import reset, uniform
import torch.nn.functional as F
# from utils.evaluate import *

EPS = 1e-15
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class GCNEncoder1(torch.nn.Module):
    '''
    GCN with k layers.
    '''
    def __init__(self, in_channels: int, out_channels: int, activation, drop_prob,
                 base_model=GCNConv,):
        '''

        :param in_channels:
        :param out_channels:
        :param activation:
        :param base_model:
        :param k: depth of base model.
        '''
        super(GCNEncoder1, self).__init__()
        self.base_model = base_model

        self.drop_prob=drop_prob
        self.conv = base_model(in_channels, out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        
        x = F.dropout(x, self.drop_prob)
        x = self.activation(self.conv(x, edge_index))
        return x

class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATEncoder, self).__init__()
        self.conv = GATConv(in_channels, out_channels,heads,)
        self.activation = nn.PReLU() #F.relu()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        return self.activation(self.conv(x, edge_index))



class HsCTRD(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, nb_nodes, embed_size,
                 nb_feature, num_project_hidden,edge_drop_prob,norm_coef):
        super(HsCTRD, self).__init__()
        # common
        self.norm_coef=norm_coef if norm_coef else 1
            
        
        self.embed_size=embed_size
        self.nb_nodes=nb_nodes
        # this is one SINGLE gcn encoder.
        fake_many_encoders = [encoder for _ in range(nb_feature)]
        # check if encoder differs
        # print([id(tt) for tt in tmp])
        
        # perturbation hyperparameter
        self.edge_drop_prob=edge_drop_prob
        
        # for intra node-node loss
        self.num_proj_hidden=num_project_hidden
        self.tau: float = 0.5
        self.fc1 = torch.nn.Linear(embed_size, num_project_hidden)
        self.fc2 = torch.nn.Linear(num_project_hidden, embed_size)

        
        # for graph-node infomax
        self.hidden_channels = hidden_channels
        self.graph_encoders = fake_many_encoders
        # summary is used for generating graph level infomation vector.
        self.summary = summary

        # for dicriminator
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

        # fusion step, graph level attention
        self.attentioned_fusion = GraphLevelAttentionLayer(embed_size, embed_size)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.graph_encoders)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)


    def forward(self, features, edge_indexes):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        # for graph
        pos_embeds = []
        neg_embeds = []
        summaries = []
        
        # encode every homogeneous graph.
        for idx,(feature, edge_index) in enumerate(zip(features, edge_indexes)):
            pos_embed = self.graph_encoders[idx](feature, edge_index)

            pos_embeds.append(pos_embed)
            # edge perturbation
            corrupted_edge,_=dropout_adj(edge_index,p=self.edge_drop_prob)

            # feature perturbation
            feat=feature.long()
            xor=torch.rand(feat.shape).to(DEVICE)
            xor[xor>0.5]=1
            xor[xor<=0.5]=0
            xor=xor.long()
            corrupted_feat=feat^xor
            corrupted_feat=corrupted_feat.float()
                
            neg_embed = self.graph_encoders[idx](corrupted_feat, corrupted_edge)
            neg_embeds.append(neg_embed)
            summary = self.summary(pos_embed, msk=None)
            summaries.append(summary)

        return pos_embeds, neg_embeds, summaries

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_zs, neg_zs, summaries):
        r"""Computes the mutal information maximization objective."""
        # graph-node infomax
        all_CSL_pos = 0
        all_CSL_neg = 0
        SCL_val=0
        for pos_z, neg_z, summary in zip(pos_zs, neg_zs, summaries):
            pos_cross_scale_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
            all_CSL_pos += pos_cross_scale_loss
            neg_cross_scale_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
            all_CSL_neg += neg_cross_scale_loss

            SCL_val+=self.same_scale_loss(pos_z, neg_z)


        CSL_val = all_CSL_pos + all_CSL_neg

        # Graph level Attention
        agged_feat=self.attentioned_fusion(torch.vstack(pos_zs))
        agged_neg = self.attentioned_fusion(torch.vstack(neg_zs))

        # Frobenius Norm
        semantic_fusion_loss=torch.sqrt(((agged_feat-agged_neg)**2).sum())
        semantic_fusion_loss=-semantic_fusion_loss

        return (SCL_val+CSL_val)+semantic_fusion_loss*self.norm_coef

    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        if type(train_z)!=np.ndarray:
            train_z=train_z.detach().cpu().numpy()
        if type(train_y)!=np.ndarray:
            train_y=train_y.detach().cpu().numpy()
        if type(test_z)!=np.ndarray:
            test_z=test_z.detach().cpu().numpy()
        if type(test_y)!=np.ndarray:
            test_y=test_y.detach().cpu().numpy()
        
        mif1s=[]
        maf1s=[]
        for i in range(10):
            clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z,
                                               train_y)
            y_pred=clf.predict(test_z)
            microf1=f1_score(test_y,y_pred,average='micro')
            macrof1=f1_score(test_y,y_pred,average='macro')
            
            mif1s.append(microf1)
            maf1s.append(macrof1)
        
        return np.mean(mif1s),np.mean(maf1s)

    def same_scale_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        # only distance between
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag()/ (between_sim.sum(1)))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)



class NodeLevelAttentionLayer(nn.Module):
    def __init__(self, embed_size,nb_features,nb_nodes):
        super(NodeLevelAttentionLayer, self).__init__()
        self.Z = nn.Parameter(torch.FloatTensor( nb_nodes, embed_size))
        self.init_weight()
        self.embed_size=embed_size
        self.nb_features=nb_features
        self.nb_nodes=nb_nodes
        self.mlps = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(nb_features)])
        self.my_coefs=None

    def forward(self, features):
        aggregated_feat, atten_coef = self.attn_feature(features)
        self.my_coefs=atten_coef
        return aggregated_feat

    def attn_feature(self, features):
        # print('I am in attention!!'+'+'*100)
        att_coef = []
        for i in range(self.nb_features):
            att_coef.append((self.mlps[i](features[i].squeeze())))
        att_coef = F.softmax(torch.cat(att_coef, 1), -1)
        features = torch.cat(features, 0)#.squeeze(0)
        attn_coef_reshaped = att_coef.transpose(1, 0).contiguous().view(-1, 1)

        aggregated_feat = features * attn_coef_reshaped.expand_as(features)
        aggregated_feat = aggregated_feat.view(self.nb_features, self.nb_nodes, self.embed_size)
        aggregated_feat = aggregated_feat.mean(dim=0)

        return aggregated_feat, att_coef

    def init_weight(self):
        nn.init.xavier_normal_(self.Z)

    def loss(self,pos,neg):
        agg_loss = F.triplet_margin_loss(self.Z, pos, neg)
        return agg_loss

class GraphLevelAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features):
        super(GraphLevelAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.my_coefs=None
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    # input (PN)*F
    def forward(self, total_embeds, P=2):

        h = torch.mm(total_embeds, self.W)
        # h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0], 1))
        # h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P, -1)
        # semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1, keepdim=True)
        # semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        self.my_coefs=semantic_attentions
        # print(semantic_attentions)
        semantic_attentions = semantic_attentions.view(P, 1, 1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_features)
        #        print(semantic_attentions)

        # input_embedding = P*N*F
        input_embedding = total_embeds.view(P, N, self.in_features)

        # h_embedding = N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        try:
            l1_loss.append(torch.abs(module.weight).sum())
        except:
            pass
        
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        try:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
        except:
            pass
    return l2_alpha * sum(l2_loss)
