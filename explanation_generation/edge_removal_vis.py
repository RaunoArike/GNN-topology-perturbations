import torch
from collections import defaultdict
import copy
import torch_geometric.utils as pyg_utils
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


def generate_perturbations(data, edges_to_remove):
    data = copy.deepcopy(data)
    edges = data.edge_index.t()
    
    edges_set = set(map(tuple, edges.numpy()))
    remove_set = set(map(tuple, edges_to_remove.numpy()))

    keep_edges = edges_set - remove_set
    keep_edges = torch.tensor(list(keep_edges)).t()
    data.edge_index = keep_edges
    
    return data


def make_undirected(lst):
    lst = torch.tensor(lst)
    flipped = lst.flip(dims=[1])
    combined = torch.cat((lst, flipped), dim=0)
    return combined


def get_logit_diff(model, data, perturbed_data):
    model.eval()
    with torch.no_grad():
        out_orig = model(data.x, data.edge_index, edge_weight=None)
        out_perturb = model(data.x, perturbed_data.edge_index, edge_weight=None)
    return calc_rrmse(out_orig, out_perturb)


def get_result_tensors(bins, model, data, num_samples, perturb_size):
    res = torch.Tensor(len(bins), num_samples)
    avg_freq = []

    for i, bin in enumerate(bins):
        avg_freq.append(np.mean([i[1] for i in bin]))
        for j in range(num_samples):
            edges = [i[0] for i in bin]
            sampled_edges = random.sample(edges, perturb_size)
            edges_to_remove = make_undirected(sampled_edges)
            perturbed_data = generate_perturbations(data, edges_to_remove)
            res[i, j] = get_logit_diff(model, data, perturbed_data)

    return res, avg_freq


def get_baseline_tensor(freq, bins, model, data, num_samples):
    res = torch.Tensor(num_samples)

    for i in range(num_samples):
        half_size = len(bins[0]) // 2
        edges = [i[0] for i in freq.items()]
        sampled_edges = random.sample(edges, half_size)
        edges_to_remove = make_undirected(sampled_edges)
        perturbed_data = generate_perturbations(data, edges_to_remove)
        res[i] = get_logit_diff(model, data, perturbed_data)

    return res


def generate_edge_removal_plots(model, data, explanations, significant_node_masks, num_bins=10, num_samples=10):
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
