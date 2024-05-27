from multiprocessing import Pool
from collections import defaultdict


def process_subset_edges(data, indices, explainer):
    edge_frequencies_big = defaultdict(int)
    edge_frequencies_small = defaultdict(int)

    for i in indices:
        print(f"here{i}")
        explanation = explainer(data.x, data.edge_index, index=i)
        edge_weight = explanation.edge_mask
        edge_index = data.edge_index

        significant_edge_mask_big = edge_weight > 0.1
        significant_edge_mask_small = edge_weight > 0.01
        significant_edge_index_big = edge_index[:, significant_edge_mask_big]
        significant_edge_index_small = edge_index[:, significant_edge_mask_small]

        for j in range(significant_edge_index_big.shape[1]):
            edge = tuple(sorted((significant_edge_index_big[0, j].item(), significant_edge_index_big[1, j].item())))
            edge_frequencies_big[edge] += 1

        for j in range(significant_edge_index_small.shape[1]):
            edge = tuple(sorted((significant_edge_index_small[0, j].item(), significant_edge_index_small[1, j].item())))
            edge_frequencies_small[edge] += 1

    return edge_frequencies_big, edge_frequencies_small


def parallel_processing_edges(data, explainer, num_processes, index_ranges):
    with Pool(processes=num_processes) as pool:
        results_big_mask, results_small_mask = pool.starmap(process_subset_edges, [(data, indices, explainer) for indices in index_ranges])
    return merge_results_edges(results_big_mask), merge_results_edges(results_small_mask)


def merge_results_edges(results):
    final_edge_frequency = {}
    for result in results:
        for edge, count in result.items():
            if edge in final_edge_frequency:
                final_edge_frequency[edge] += count
            else:
                final_edge_frequency[edge] = 1
    return final_edge_frequency


def get_top_edges(edge_frequency, top_n=10):
    sorted_edges = sorted(edge_frequency.items(), key=lambda item: item[1], reverse=True)
    top_edges = sorted_edges[:top_n]
    return top_edges
