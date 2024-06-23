import torch
import csv
import argparse
import ast

from utils import train_model
from node_removal_vis import generate_node_removal_plots
from edge_removal_vis import generate_edge_removal_plots
from edge_weights_vis import generate_edge_weights_plots
from plotting import plot_with_confidence_intervals, plot_baseline
from load_data import load_cora, load_DBLP
from gat import GAT
from gcn import GCN

import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--task", type=str, default="node_removal")

    args = parser.parse_args()
    dataset_type = args.dataset
    task = args.task

    if dataset_type == "Cora":
        dataset, data = load_cora()
    elif dataset_type == "DBLP":
        dataset, data = load_DBLP()
    else:
        raise Exception("Support for other datasets has not been implemented yet!")

    GCN_conf = {
        "num_features": dataset.num_features if dataset_type == "Cora" else dataset.num_features['author'],
        "num_classes": dataset.num_classes if dataset_type == "Cora" else data.y.max().item() + 1,
        "hidden_channels": 16,
        "learning_rate": 0.01,
        "weight_decay": 5e-4
    }
    GCN_model = GCN(GCN_conf)

    GAT_conf = {
        "num_features": dataset.num_features if dataset_type == "Cora" else dataset.num_features['author'],
        "num_classes": dataset.num_classes if dataset_type == "Cora" else data.y.max().item() + 1,
        "hidden_channels": 8,
        "num_heads": 8,
        "learning_rate": 0.005,
        "weight_decay": 5e-4
    }
    GAT_model = GAT(GAT_conf)
    
    GCN_opt = torch.optim.Adam(GCN_model.parameters(), lr=GCN_conf["learning_rate"], weight_decay=GCN_conf["weight_decay"])
    GAT_opt = torch.optim.Adam(GAT_model.parameters(), lr=GAT_conf["learning_rate"], weight_decay=GAT_conf["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()

    train_model(GCN_model, data, GCN_opt, loss_fn)
    train_model(GAT_model, data, GAT_opt, loss_fn)

    GCN_GNNExplainer_explanations = {}
    GCN_IG_explanations = {}
    GAT_GNNExplainer_explanations = {}
    GAT_IG_explanations = {}

    try:
        if task == "node_removal":
            with open(f"../results/Explanations_GCN_GNNExplainer_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        node_indices = row[1]
                        node_indices_list = ast.literal_eval(node_indices)
                        node_weights = row[2]
                        node_weights_list = ast.literal_eval(node_weights)
                        GCN_GNNExplainer_explanations[index] = (node_indices_list, node_weights_list)

            with open(f"../results/Explanations_GCN_IntegratedGradients_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        node_indices = row[1]
                        node_indices_list = ast.literal_eval(node_indices)
                        node_weights = row[2]
                        node_weights_list = ast.literal_eval(node_weights)
                        GCN_IG_explanations[index] = (node_indices_list, node_weights_list)

            with open(f"../results/Explanations_GAT_GNNExplainer_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        node_indices = row[1]
                        node_indices_list = ast.literal_eval(node_indices)
                        node_weights = row[2]
                        node_weights_list = ast.literal_eval(node_weights)
                        GAT_GNNExplainer_explanations[index] = (node_indices_list, node_weights_list)

            with open(f"../results/Explanations_GAT_IntegratedGradients_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        node_indices = row[1]
                        node_indices_list = ast.literal_eval(node_indices)
                        node_weights = row[2]
                        node_weights_list = ast.literal_eval(node_weights)
                        GAT_IG_explanations[index] = (node_indices_list, node_weights_list)
        else:
            with open(f"../results/Explanations_GCN_GNNExplainer_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        edge_indices = row[3]
                        edge_indices_list = ast.literal_eval(edge_indices)
                        edge_weights = row[4]
                        edge_weights_list = ast.literal_eval(edge_weights)
                        GCN_GNNExplainer_explanations[index] = (edge_indices_list, edge_weights_list)

            with open(f"../results/Explanations_GCN_IntegratedGradients_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        edge_indices = row[3]
                        edge_indices_list = ast.literal_eval(edge_indices)
                        edge_weights = row[4]
                        edge_weights_list = ast.literal_eval(edge_weights)
                        GCN_IG_explanations[index] = (edge_indices_list, edge_weights_list)

            with open(f"../results/Explanations_GAT_GNNExplainer_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        edge_indices = row[3]
                        edge_indices_list = ast.literal_eval(edge_indices)
                        edge_weights = row[4]
                        edge_weights_list = ast.literal_eval(edge_weights)
                        GAT_GNNExplainer_explanations[index] = (edge_indices_list, edge_weights_list)

            with open(f"../results/Explanations_GAT_IntegratedGradients_{dataset_type}.csv", mode='r') as infile:
                reader = csv.reader(infile)
                next(reader)
                for row in reader:
                    if row:
                        index = row[0]
                        edge_indices = row[3]
                        edge_indices_list = ast.literal_eval(edge_indices)
                        edge_weights = row[4]
                        edge_weights_list = ast.literal_eval(edge_weights)
                        GAT_IG_explanations[index] = (edge_indices_list, edge_weights_list)
    except Exception:
        raise Exception("The data file for the required model, explainer and dataset combination doesn't exist yet!")
    
    if task == "node_removal":
        GCN_GNNExplainer_res_big, GCN_GNNExplainer_res_small, GCN_GNNExplainer_avg_freq_big, GCN_GNNExplainer_avg_freq_small, GCN_baseline_big, GCN_baseline_small = generate_node_removal_plots(GCN_model, data, GCN_GNNExplainer_explanations, (0.3, 0.1))
        GCN_IG_res_big, GCN_IG_res_small, GCN_IG_avg_freq_big, GCN_IG_avg_freq_small, _, _ = generate_node_removal_plots(GCN_model, data, GCN_IG_explanations, (0.003, 0.0001))
        GAT_GNNExplainer_res_big, GAT_GNNExplainer_res_small, GAT_GNNExplainer_avg_freq_big, GAT_GNNExplainer_avg_freq_small, GAT_baseline_big, GAT_baseline_small = generate_node_removal_plots(GAT_model, data, GAT_GNNExplainer_explanations, (0.3, 0.1))
        GAT_IG_res_big, GAT_IG_res_small, GAT_IG_avg_freq_big, GAT_IG_avg_freq_small, _, _ = generate_node_removal_plots(GAT_model, data, GAT_IG_explanations, (0.003, 0.0001))
    elif task == "edge_removal":
        GCN_GNNExplainer_res_big, GCN_GNNExplainer_res_small, GCN_GNNExplainer_avg_freq_big, GCN_GNNExplainer_avg_freq_small, GCN_baseline_big, GCN_baseline_small = generate_edge_removal_plots(GCN_model, data, GCN_GNNExplainer_explanations, (0.3, 0.1))
        GCN_IG_res_big, GCN_IG_res_small, GCN_IG_avg_freq_big, GCN_IG_avg_freq_small, _, _ = generate_edge_removal_plots(GCN_model, data, GCN_IG_explanations, (0.003, 0.0001))
        GAT_GNNExplainer_res_big, GAT_GNNExplainer_res_small, GAT_GNNExplainer_avg_freq_big, GAT_GNNExplainer_avg_freq_small, GAT_baseline_big, GAT_baseline_small = generate_edge_removal_plots(GAT_model, data, GAT_GNNExplainer_explanations, (0.3, 0.1))
        GAT_IG_res_big, GAT_IG_res_small, GAT_IG_avg_freq_big, GAT_IG_avg_freq_small, _, _ = generate_edge_removal_plots(GAT_model, data, GAT_IG_explanations, (0.003, 0.0001))
    elif task == "edge_weights":
        GCN_GNNExplainer_res_big, GCN_GNNExplainer_res_small, GCN_GNNExplainer_avg_freq_big, GCN_GNNExplainer_avg_freq_small, GCN_baseline_big, GCN_baseline_small = generate_edge_removal_plots(GCN_model, data, GCN_GNNExplainer_explanations, (0.3, 0.1))
        GCN_IG_res_big, GCN_IG_res_small, GCN_IG_avg_freq_big, GCN_IG_avg_freq_small, _, _ = generate_edge_removal_plots(GCN_model, data, GCN_IG_explanations, (0.003, 0.0001))
        GAT_GNNExplainer_res_big, GAT_GNNExplainer_res_small, GAT_GNNExplainer_avg_freq_big, GAT_GNNExplainer_avg_freq_small, GAT_baseline_big, GAT_baseline_small = generate_edge_removal_plots(GAT_model, data, GAT_GNNExplainer_explanations, (0.3, 0.1))
        GAT_IG_res_big, GAT_IG_res_small, GAT_IG_avg_freq_big, GAT_IG_avg_freq_small, _, _ = generate_edge_removal_plots(GAT_model, data, GAT_IG_explanations, (0.003, 0.0001))
    else:
        raise Exception("Support for this type of perturbation hasn't been implemented yet!")

    plt.figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

    plot_with_confidence_intervals(GCN_GNNExplainer_avg_freq_small, GCN_GNNExplainer_res_small.numpy(), 'GCN + GNNExplainer', colors[0])
    plot_with_confidence_intervals(GCN_IG_avg_freq_small, GCN_IG_res_small.numpy(), 'GCN + Integrated Gradients', colors[1])
    plot_with_confidence_intervals(GAT_GNNExplainer_avg_freq_small, GAT_GNNExplainer_res_small.numpy(), 'GAT + GNNExplainer', colors[2])
    plot_with_confidence_intervals(GAT_IG_avg_freq_small, GAT_IG_res_small.numpy(), 'GAT + Integrated Gradients', colors[3])
    plot_baseline(GCN_baseline_small.numpy(), "GCN Baseline", colors[4])
    plot_baseline(GAT_baseline_small.numpy(), "GAT Baseline", colors[5])

    plt.legend()

    if task == "node_removal":
        plt.xlabel('Mean number of times the nodes in the frequency group\nappeared on explanation subgraphs')
        plt.ylabel('RRMSE')
        plt.title(f'Stability to node removal perturbations as a function of\nthe frequency with which the nodes appear on\nexplanation subgraphs, small $\\gamma$, {dataset_type} dataset')
    elif task == "edge_removal":
        plt.xlabel('Mean number of times the edges in the frequency group\nappeared on explanation subgraphs')
        plt.ylabel('RRMSE')
        plt.title(f'Stability to edge removal perturbations as a function of\nthe frequency with which the nodes appear on\nexplanation subgraphs, small $\\gamma$, {dataset_type} dataset')
    else:
        plt.xlabel('Mean number of times the edges in the frequency group\nappeared on explanation subgraphs')
        plt.ylabel('RRMSE')
        plt.title(f'Stability to edge weight perturbations as a function of\nthe frequency with which the nodes appear on\nexplanation subgraphs, small $\\gamma$, {dataset_type} dataset')

    plt.savefig(f'../img/res_{task}_small_{dataset_type}.png', bbox_inches='tight')

    plt.figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

    plot_with_confidence_intervals(GCN_GNNExplainer_avg_freq_big, GCN_GNNExplainer_res_big.numpy(), 'GCN + GNNExplainer', colors[0])
    plot_with_confidence_intervals(GCN_IG_avg_freq_big, GCN_IG_res_big.numpy(), 'GCN + Integrated Gradients', colors[1])
    plot_with_confidence_intervals(GAT_GNNExplainer_avg_freq_big, GAT_GNNExplainer_res_big.numpy(), 'GAT + GNNExplainer', colors[2])
    plot_with_confidence_intervals(GAT_IG_avg_freq_big, GAT_IG_res_big.numpy(), 'GAT + Integrated Gradients', colors[3])
    plot_baseline(GCN_baseline_big.numpy(), "GCN Baseline", colors[4])
    plot_baseline(GAT_baseline_big.numpy(), "GAT Baseline", colors[5])

    plt.legend()
    if task == "node_removal":
        plt.xlabel('Mean number of times the nodes in the frequency group\nappeared on explanation subgraphs')
        plt.ylabel('RRMSE')
        plt.title(f'Stability to node removal perturbations as a function of\nthe frequency with which the nodes appear on\nexplanation subgraphs, large $\\gamma$, {dataset_type} dataset')
    elif task == "edge_removal":
        plt.xlabel('Mean number of times the edges in the frequency group\nappeared on explanation subgraphs')
        plt.ylabel('RRMSE')
        plt.title(f'Stability to edge removal perturbations as a function of\nthe frequency with which the nodes appear on\nexplanation subgraphs, large $\\gamma$, {dataset_type} dataset')
    else:
        plt.xlabel('Mean number of times the edges in the frequency group\nappeared on explanation subgraphs')
        plt.ylabel('RRMSE')
        plt.title(f'Stability to edge weight perturbations as a function of\nthe frequency with which the nodes appear on\nexplanation subgraphs, large $\\gamma$, {dataset_type} dataset')

    plt.savefig(f'../img/res_{task}_big_{dataset_type}.png', bbox_inches='tight')
