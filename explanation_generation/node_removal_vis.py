import torch
from collections import defaultdict
import copy
import torch_geometric.utils as pyg_utils
import numpy as np
import random
from utils import divide_into_chunks

from utils import calc_rrmse


def get_node_frequencies(explanations, significant_node_masks):
    node_frequencies_big = defaultdict(int)
    node_frequencies_small = defaultdict(int)

    for i, (node_indices, node_weights) in explanations.items():
        node_indices = torch.tensor(node_indices)
        node_weights = torch.tensor(node_weights)

        significant_node_mask_big = node_weights > significant_node_masks[0]
        significant_node_mask_small = node_weights > significant_node_masks[1]
        significant_nodes_big = node_indices[significant_node_mask_big]
        significant_nodes_small = node_indices[significant_node_mask_small]

        significant_nodes_big = np.unique(significant_nodes_big.numpy())
        significant_nodes_small = np.unique(significant_nodes_small.numpy())

        for node in significant_nodes_big:
            node_frequencies_big[node] += 1

        for node in significant_nodes_small:
            node_frequencies_small[node] += 1

    return node_frequencies_big, node_frequencies_small


def generate_perturbations(data, nodes_to_remove):
    data = copy.deepcopy(data)
    nodes_to_remove = set(nodes_to_remove)

    mask = torch.ones(data.x.size(0), dtype=torch.bool)
    mask[list(nodes_to_remove)] = False
    
    data.x = data.x[mask]

    all_nodes = set(range(data.num_nodes))
    nodes_to_retain = all_nodes.difference(nodes_to_remove)
    data.edge_index = pyg_utils.subgraph(sorted(nodes_to_retain), data.edge_index, relabel_nodes=True)[0]

    return data, mask


def get_logit_diff(data, perturbed_data, model, mask):
    model.eval()
    with torch.no_grad():
        out_orig = model(data.x, data.edge_index, edge_weight=None)
        out_perturb = model(perturbed_data.x, perturbed_data.edge_index, edge_weight=None)
    return calc_rrmse(out_orig[mask], out_perturb)


def get_result_tensors(bins, model, data, num_samples, perturb_size):
    avg_freq = []
    res = torch.zeros(len(bins), num_samples)

    for i, bin in enumerate(bins):
        avg_freq.append(np.mean([i[1] for i in bin]))
        for j in range(num_samples):
            nodes = [i[0] for i in bin]
            sampled_nodes = random.sample(nodes, perturb_size)
            perturbed_data, mask = generate_perturbations(data, sampled_nodes)
            res[i, j] = get_logit_diff(data, perturbed_data, model, mask)

    return res, avg_freq


def get_baseline_tensor(freq, bins, model, data, num_samples):
    res = torch.zeros(num_samples)

    for i in range(num_samples):
        half_size = len(bins[0]) // 2
        nodes = [i[0] for i in freq.items()]
        sampled_nodes = random.sample(nodes, half_size)
        perturbed_data, mask = generate_perturbations(data, sampled_nodes)
        res[i] = get_logit_diff(data, perturbed_data, model, mask)

    return res


def generate_node_removal_plots(model, data, explanations, significant_node_masks, num_bins=10, num_samples=10):
    node_freq_big, node_freq_small = get_node_frequencies(explanations, significant_node_masks)
    bins_big = divide_into_chunks(node_freq_big, num_bins)
    bins_small = divide_into_chunks(node_freq_small, num_bins)
    perturb_size_big = len(bins_big[0]) // 2
    perturb_size_small = len(bins_small[0]) // 2
    res_big, avg_freq_big = get_result_tensors(bins_big, model, data, num_samples, perturb_size_big)
    res_small, avg_freq_small = get_result_tensors(bins_small, model, data, num_samples, perturb_size_small)
    baseline_big = get_baseline_tensor(node_freq_big, bins_big, model, data, num_samples)
    baseline_small = get_baseline_tensor(node_freq_small, bins_small, model, data, num_samples)
    return res_big, res_small, avg_freq_big, avg_freq_small, baseline_big, baseline_small
