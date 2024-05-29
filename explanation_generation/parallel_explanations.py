import torch
import numpy as np
import csv
from collections import defaultdict
from multiprocessing import Pool
import argparse

from explainer import initialize_GNNExplainer, initialize_IGExplainer
from load_data import load_cora
from gat import GAT, train_GAT, test_GAT
from gcn import GCN, train_GCN, test_GCN
# from explanation_generation.tagcn import TAGCN, train, test


def get_explanations(data, indices, explainer):
    res = []

    for i in indices:
        print(f"here{i}")
        explanation = explainer(data.x, data.edge_index, index=int(i))
        node_mask = explanation.node_mask.detach()
        edge_mask = explanation.edge_mask.detach()
        res.append((i, node_mask, edge_mask))

    return res


def generate_explanations(explainer):
    num_processes = 10
    index_ranges = np.array_split(range(data.num_nodes), num_processes)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(get_explanations, [(data, indices, explainer) for indices in index_ranges])

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
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    elif isinstance(model, GAT):
        for epoch in range(1, 101):
            loss = train_GAT(model, data, optimizer, loss_fn)
            val_acc = test_GAT(model, data, data.val_mask)
            test_acc = test_GAT(model, data, data.test_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--explainer", type=str, default="GNNExplainer")

    args = parser.parse_args()
    model_type = args.model
    explainer_type = args.explainer

    dataset, data = load_cora()
    print(model_type)
    if model_type == "GCN":
        conf = {
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "hidden_channels": 16,
            "learning_rate": 0.01,
            "weight_decay": 5e-4
        }
        model = GCN(conf)

    elif model_type == "GAT":
        conf = {
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "hidden_channels": 8,
            "num_heads": 8,
            "learning_rate": 0.005,
            "weight_decay": 5e-4
        }
        model = GAT(conf)

    else:
        raise Exception("Support for other models has not been implemented yet!")
    
    opt = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()

    train_model(model, data, opt, loss_fn)

    if explainer_type == "GNNExplainer":
        explainer = initialize_GNNExplainer(model, epochs=100)
    elif explainer_type == "IntegratedGradients":
        explainer = initialize_IGExplainer(model)
    else:
        raise Exception("Support for other explainers has not been implemented yet!")
    
    res = generate_explanations(explainer)

    with open(f"../Results/Explanations_{model_type}_{explainer_type}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Explained node", "Nodes with nonzero weights", "Node weights", "Edges with nonzero weights", "Edge weights"])
        for node, node_mask, edge_mask in res:
            nonzero_node_weight_indices = torch.where(node_mask > 0)
            node_weights = node_mask[nonzero_node_weight_indices]
            nonzero_edge_weight_indices = torch.where(edge_mask > 0)
            edge_weights = edge_mask[nonzero_edge_weight_indices]
            writer.writerow([node, nonzero_node_weight_indices[0].tolist(), node_weights.tolist(), nonzero_edge_weight_indices[0].tolist(), edge_weights.tolist()])
    