import numpy as np
from multiprocessing import Pool
import torch

from gat import GAT, train_GAT, test_GAT
from gcn import GCN, train_GCN, test_GCN


def get_explanations(model, data, indices, explainer):
    res = []

    for i in indices:
        print(f"here{i}")
        explanation = explainer(data.x, data.edge_index, edge_weight=data.edge_weight, index=int(i))
        node_mask = explanation.node_mask.detach()
        edge_mask = explanation.edge_mask.detach()
        res.append((i, node_mask, edge_mask))

    return res


def generate_explanations(model, explainer, data):
    num_processes = 10
    index_ranges = np.array_split(range(data.num_nodes), num_processes)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(get_explanations, [(model, data, indices, explainer) for indices in index_ranges])

    merged_res = []
    for res in results:
        merged_res = merged_res + res

    return sorted(merged_res, key=lambda x: x[0])



def train_model(model, data, optimizer, loss_fn):
    if isinstance(model, GCN):
        for epoch in range(1, 101):
            loss = train_GCN(model, data, optimizer, loss_fn)
            val_acc = test_GCN(model, data, data.val_mask)
            test_acc = test_GCN(model, data, data.test_mask)
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    elif isinstance(model, GAT):
        for epoch in range(1, 101):
            loss = train_GAT(model, data, optimizer, loss_fn)
            val_acc = test_GAT(model, data, data.val_mask)
            test_acc = test_GAT(model, data, data.test_mask)
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')



def divide_into_chunks(freq, n):
    lst = sorted(freq.items(), key=lambda x: x[1])
    """Divide the list lst into n equally-sized chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def calc_rrmse(y, y_hat):
    residuals = y - y_hat
    squared_norm_residuals = torch.linalg.vector_norm(residuals)**2
    squared_norm_orig = torch.linalg.vector_norm(y_hat)**2
    return torch.sqrt(squared_norm_residuals / squared_norm_orig)
