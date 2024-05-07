from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_cora():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]  # There's only one graph in the dataset - get the first graph object.
    return dataset, data
