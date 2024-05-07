from multiprocessing import Pool


def process_subset(data, indices, explainer):
    edge_frequency = {}
    for i in indices:
        print("here")
        explanation = explainer(data.x, data.edge_index, index=i)
        print("here1")
        edge_weight = explanation.edge_mask
        edge_index = data.edge_index

        significant_edge_mask = edge_weight > 0.5
        significant_edge_index = edge_index[:, significant_edge_mask]

        for j in range(significant_edge_index.shape[1]):
            edge = tuple(sorted((significant_edge_index[0, j].item(), significant_edge_index[1, j].item())))
            
            if edge in edge_frequency:
                edge_frequency[edge] += 1
            else:
                edge_frequency[edge] = 1
    return edge_frequency


def parallel_processing(data, explainer, num_processes, index_ranges):
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_subset, [(data, indices, explainer) for indices in index_ranges])
    return merge_results(results)


def merge_results(results):
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
