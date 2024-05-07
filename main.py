import torch
import sys
from torch_geometric.explain import GNNExplainer
import numpy as np

from explainer import initialize_explainer
from load_data import load_cora
from parallel import parallel_processing, get_top_edges
from gat import GAT, train, test
from gcn import GCN, train, test


if __name__ == "__main__":    
    dataset, data = load_cora()
    # conf = {
    #     "num_features": dataset.num_features,
    #     "num_classes": dataset.num_classes,
    #     "num_heads": 8,
    #     "hidden_channels": 8
    # }
    conf = {
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "hidden_channels": 16
    }

    # model = GAT(conf)
    model = GCN(conf)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        loss = train(model, data, optimizer, loss_fn)
        val_acc = test(model, data, data.val_mask)
        test_acc = test(model, data, data.test_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    explainer = initialize_explainer(model, GNNExplainer(epochs=200), conf)
    
    num_processes = 10
    index_ranges = np.array_split(range(30), num_processes)
    freq = parallel_processing(data, explainer, num_processes, index_ranges)
    res = get_top_edges(freq)

    print(res)
