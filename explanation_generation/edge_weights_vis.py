import torch
from collections import defaultdict
import copy
import numpy as np
import random

from utils import calc_rrmse, divide_into_chunks


def get_edge_frequencies(data, explanations, significant_node_masks):
    edge_frequencies_big = defaultdict(int)
    edge_frequencies_small = defaultdict(int)

    for i, (edge_indices, edge_weights) in explanations.items():
        edge_indices = torch.tensor(edge_indices)
        edge_weights = torch.tensor(edge_weights)

        significant_edge_mask_big = edge_indices[edge_weights > significant_node_masks[0]].to(torch.int64)
        significant_edge_mask_small = edge_indices[edge_weights > significant_node_masks[1]].to(torch.int64)

        significant_edge_index_big = data.edge_index[:, significant_edge_mask_big]
        significant_edge_index_small = data.edge_index[:, significant_edge_mask_small]

        for j in range(significant_edge_index_big.shape[1]):
            edge = tuple(sorted((significant_edge_index_big[0, j].item(), significant_edge_index_big[1, j].item())))
            edge_frequencies_big[edge] += 1

        for j in range(significant_edge_index_small.shape[1]):
            edge = tuple(sorted((significant_edge_index_small[0, j].item(), significant_edge_index_small[1, j].item())))
            edge_frequencies_small[edge] += 1

    return edge_frequencies_big, edge_frequencies_small


def generate_edge_weight_perturbations(data, edges_to_perturb, max_change=20):
    data = copy.deepcopy(data)
    edges = data.edge_index.t()
    edge_weights = data.edge_weight.clone()
    
    edges_to_perturb_set = set(map(tuple, edges_to_perturb.numpy()))

    perturbed_indices = []
    perturbed_weights = []

    # Identify perturbed edges and store their original weights
    for idx, edge in enumerate(edges):
        if tuple(edge.numpy()) in edges_to_perturb_set:
            perturbed_indices.append(idx)
            perturbed_weights.append(edge_weights[idx])

    # Compute original norm of perturbed weights
    original_norm = torch.norm(torch.tensor(perturbed_weights), p=2)

    # Apply perturbations
    for idx in perturbed_indices:
        change = max_change * (2 * torch.rand(1) - 1)  # Uniform distribution from -max_change to +max_change
        new_weight = edge_weights[idx] + change
        edge_weights[idx] = abs(new_weight)  # Ensure no negative weights

    # Calculate norm of perturbed weights
    perturbed_new_weights = edge_weights[perturbed_indices]
    current_norm = torch.norm(perturbed_new_weights, p=2)

    # Renormalize only perturbed edge weights
    renormalization_factor = original_norm / current_norm
    edge_weights[perturbed_indices] *= renormalization_factor

    print(torch.norm(edge_weights))

    data.edge_weight = edge_weights
    
    return data


def make_undirected(lst):
    lst = torch.tensor(lst)
    flipped = lst.flip(dims=[1])
    combined = torch.cat((lst, flipped), dim=0)
    return combined


def get_logit_diff(model, data, perturbed_data):
    model.eval()
    with torch.no_grad():
        out_orig = model(data.x, data.edge_index)
        out_perturb = model(data.x, perturbed_data.edge_index)
    return calc_rrmse(out_orig, out_perturb)


def get_result_tensors(bins, model, data, num_samples, perturb_size):
    res = torch.Tensor(len(bins), num_samples)
    avg_freq = []

    for i, bin in enumerate(bins):
        for j in range(num_samples):
            half_size = len(bin) // 2
            edges = [i[0] for i in bin]
            sampled_edges = random.sample(edges, half_size)
            edges_to_remove = make_undirected(sampled_edges)
            perturbed_data = generate_edge_weight_perturbations(data, edges_to_remove)
            res[i, j] = get_logit_diff(model, data, perturbed_data)

    return res, avg_freq


def get_baseline_tensor(freq, bins, model, data, num_samples):
    res = torch.Tensor(num_samples)

    for i in range(num_samples):
        half_size = len(bins[0]) // 2
        edges = [i[0] for i in freq.items()]
        sampled_edges = random.sample(edges, half_size)
        edges_to_remove = make_undirected(sampled_edges)
        perturbed_data = generate_edge_weight_perturbations(data, edges_to_remove)
        res[i] = get_logit_diff(model, data, perturbed_data)

    return res


def generate_edge_weights_plots(model, data, explanations, significant_node_masks, num_bins=10, num_samples=10):
    edge_freq_big, edge_freq_small = get_edge_frequencies(data, explanations, significant_node_masks)
    bins_big = divide_into_chunks(edge_freq_big, num_bins)
    bins_small = divide_into_chunks(edge_freq_small, num_bins)
    perturb_size_big = len(bins_big[0]) // 2
    perturb_size_small = len(bins_small[0]) // 2
    res_big, avg_freq_big = get_result_tensors(bins_big, model, data, num_samples, perturb_size_big)
    res_small, avg_freq_small = get_result_tensors(bins_small, model, data, num_samples, perturb_size_small)
    baseline_big = get_baseline_tensor(edge_freq_big, bins_big, model, data, num_samples)
    baseline_small = get_baseline_tensor(edge_freq_small, bins_small, model, data, num_samples)
    return res_big, res_small, avg_freq_big, avg_freq_small, baseline_big, baseline_small
