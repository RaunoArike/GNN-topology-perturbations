import torch
import csv
import argparse

from utils import train_model, generate_explanations
from explainer import initialize_GNNExplainer, initialize_CaptumExplainer
from load_data import load_cora, load_DBLP
from gat import GAT
from gcn import GCN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--explainer", type=str, default="GNNExplainer")
    parser.add_argument("--dataset", type=str, default="Cora")

    args = parser.parse_args()
    model_type = args.model
    explainer_type = args.explainer
    dataset_type = args.dataset

    if dataset_type == "Cora":
        dataset, data = load_cora()
    elif dataset_type == "DBLP":
        dataset, data = load_DBLP()
    else:
        raise Exception("Support for other datasets has not been implemented yet!")

    if model_type == "GCN":
        conf = {
            "num_features": dataset.num_features if dataset_type == "Cora" else dataset.num_features['author'],
            "num_classes": dataset.num_classes if dataset_type == "Cora" else data.y.max().item() + 1,
            "hidden_channels": 16,
            "learning_rate": 0.01,
            "weight_decay": 5e-4
        }
        model = GCN(conf)

    elif model_type == "GAT":
        conf = {
            "num_features": dataset.num_features if dataset_type == "Cora" else dataset.num_features['author'],
            "num_classes": dataset.num_classes if dataset_type == "Cora" else data.y.max().item() + 1,
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
    elif explainer_type == "IntegratedGradients" or explainer_type == "Saliency" or explainer_type == "InputXGradient" or explainer_type == "Deconvolution" or explainer_type == "GuidedBackprop":
        explainer = initialize_CaptumExplainer(model, explainer_type)
    else:
        raise Exception("Support for other explainers has not been implemented yet!")
    
    res = generate_explanations(model, explainer, data)

    with open(f"../Results/Explanations_{model_type}_{explainer_type}_{dataset_type}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Explained node", "Nodes with nonzero weights", "Node weights", "Edges with nonzero weights", "Edge weights"])
        for node, node_mask, edge_mask in res:
            nonzero_node_weight_indices = torch.where(node_mask > 0)
            node_weights = node_mask[nonzero_node_weight_indices]
            nonzero_edge_weight_indices = torch.where(edge_mask > 0)
            edge_weights = edge_mask[nonzero_edge_weight_indices]
            writer.writerow([node, nonzero_node_weight_indices[0].tolist(), node_weights.tolist(), nonzero_edge_weight_indices[0].tolist(), edge_weights.tolist()])
    