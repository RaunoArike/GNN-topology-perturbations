import torch
import sys
from torch_geometric.explain import GNNExplainer
import numpy as np
import csv

from explanation_generation.explainer import initialize_explainer
from explanation_generation.load_data import load_cora
from deprecated.parallel_edges import parallel_processing_edges, get_top_edges
from explanation_generation.gat import GAT, train, test
from explanation_generation.gcn import GCN, train, test


if __name__ == "__main__":    
    dataset, data = load_cora()
    conf = {
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "num_heads": 8,
        "hidden_channels": 8
    }
    # conf = {
    #     "num_features": dataset.num_features,
    #     "num_classes": dataset.num_classes,
    #     "hidden_channels": 16
    # }

    model = GAT(conf)
    # model = GCN(conf)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(model, data, optimizer, loss_fn)
        val_acc = test(model, data, data.val_mask)
        test_acc = test(model, data, data.test_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    explainer = initialize_explainer(model, GNNExplainer(epochs=200), conf)
    
    num_processes = 10
    index_ranges = np.array_split(range(data.num_nodes), num_processes)
    freq = parallel_processing_edges(data, explainer, num_processes, index_ranges)
    sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Edge", "Frequency"])
        for edge, frequency in freq.items():
            writer.writerow([f"{edge[0]}-{edge[1]}", frequency])
    