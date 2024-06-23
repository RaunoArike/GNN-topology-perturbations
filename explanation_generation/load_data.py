from torch_geometric.datasets import Planetoid, DBLP
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from collections import defaultdict
import torch


def load_cora():
    dataset = Planetoid(root='../data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]  # There's only one graph in the dataset - get the first graph object.
    return dataset, data

def load_DBLP():
    dataset = DBLP(root='../data/dblp', transform=NormalizeFeatures())
    hetero_data = dataset[0]
    data = convert_to_weighted_homog_graph(hetero_data)
    return dataset, data


def convert_to_weighted_homog_graph(hetero_data):    
    # Initialize adjacency list to count co-authorship
    adj_dict = defaultdict(int)
    
    author_to_paper = hetero_data['author', 'to', 'paper'].edge_index
    paper_to_author = hetero_data['paper', 'to', 'author'].edge_index
    
    for author, paper in author_to_paper.t().tolist():
        coauthors = paper_to_author[1, paper_to_author[0] == paper].tolist()
        for coauthor in coauthors:
            if author != coauthor:
                edge = tuple([author, coauthor])
                adj_dict[edge] += 1
    
    edge_index = torch.tensor(list(adj_dict.keys()), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(list(adj_dict.values()), dtype=torch.float)
    
    data = Data()
    data.x = hetero_data['author'].x
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    data.train_mask = hetero_data['author'].train_mask
    data.val_mask = hetero_data['author'].val_mask
    data.test_mask = hetero_data['author'].test_mask
    data.y = hetero_data['author'].y
    
    return data
