from torch_geometric.explain import Explainer


def initialize_explainer(model, explainer_type, conf):
    explainer = Explainer(
        model=model,
        algorithm=explainer_type,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    return explainer
