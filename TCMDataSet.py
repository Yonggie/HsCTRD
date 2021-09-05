import torch
import pickle
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import scipy.sparse as sp
import networkx as nx


class TCMDataSet(InMemoryDataset):
    '''
    edge_index： all edge in graph
    '''

    def __init__(self, root, dataset_name, feature_size, metapath_name, transform=None, pre_transform=None,
                 show=False):
        self.show = show
        print(f'feature size: {feature_size}')

        super(TCMDataSet, self).__init__(root, transform, pre_transform)

        base = f"datasets/{dataset_name}/"

        edge_idx_dict=pickle.load(open(base+'meta_edge_dict.pkl','rb'))

        edge_idxes=edge_idx_dict[metapath_name]
        # full labels.
        labels=pickle.load(open(base+'labels.pkl','rb'))
        # labeled_idx=pickle.load(open(base+'labeled_idx.pkl','rb'))
        # feature = torch.randn(len(labels),feature_size)
        feature = pickle.load(open(base + 'feature.pkl', 'rb'))


        data = Data(
            x=[feature for _ in edge_idxes],
            edge_indexes=[item.t().contiguous() for item in edge_idxes],
            y=labels,
        )

        self.data = data

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        return

    @staticmethod
    def show_graph(net):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.from_edgelist(net)
        nx.draw(G, pos=nx.spring_layout(G),
                node_color='b',
                edge_color='r',
                with_labels=False,
                font_size=1,
                node_size=2)
        plt.show()

    def process(self):
        return