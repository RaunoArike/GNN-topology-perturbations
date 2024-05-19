import torch
import sys
from torch_geometric.explain import GNNExplainer
import numpy as np
import csv

from explainer import initialize_explainer
from load_data import load_cora
from parallel_edges import parallel_processing_edges, get_top_edges
# from gat import GAT, train, test
from gcn import GCN, train, test
from multiprocessing import Pool


def get_explanations(data, indices, explainer):
    res = {}

    for i in indices:
        print(f"here{i}")
        explanation = explainer(data.x, data.edge_index, index=i)
        edge_weight = explanation.edge_mask
        res[i] = edge_weight

    return res


if __name__ == "__main__":    
    dataset, data = load_cora()
    conf = {
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "hidden_channels": 16
    }

    model = GCN(conf)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    # loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(model, data, optimizer, loss_fn)
        val_acc = test(model, data, data.val_mask)
        test_acc = test(model, data, data.test_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    explainer = initialize_explainer(model, GNNExplainer(epochs=100), conf)

    num_processes = 10
    index_ranges = np.array_split(range(data.num_nodes), num_processes)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(get_explanations, [(data, indices, explainer) for indices in index_ranges])

    merged_res = {}
    for res in results:
        merged_res.update(res)

    with open("explanations_gcn.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Explained node", "Edge weights"])
        for node, explanation in merged_res.items():
            writer.writerow([node, explanation.tolist()])
