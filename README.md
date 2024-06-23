# Beyond Spectral Graph Theory: An Explainability-Driven Approach to Analyzing the Stability of GNNs to Topology Perturbations

This is the repository for a bachelor's thesis titled "Beyond Spectral Graph Theory: An Explainability-Driven Approach to Analyzing the Stability of GNNs to Topology Perturbations". The thesis has been written by Rauno Arike as part of the CSE3000 Research Project course. The paper that the code in this repository accompanies can be found at ...

## Reproducing the results

The raw data that has been used to generate the plots shown in the paper can be found from the `/results` folder. To reproduce our results, the following steps should be followed:

1. Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Reproduce the results

The commands listed here should be run by anyone who wants to reproduce the raw data that was used to generate our plots or to generate results for additional explainers. In the paper, we present results for the GNNExplainer and Integrated Gradients explainability tools, but support has also been implemented for Saliency Attribution, InputXGradient, Deconvolution, and Guided Backpropagation.

```
cd explanation_generation
generate_explanations.py --model "GCN" --explainer "GNNExplainer" --dataset "Cora"
# supported arguments for model: "GCN", "GAT"
# supported arguments for explainer: "GNNExplainer", "IntegratedGradients", "Saliency", "InputXGradient", "Deconvolution", "GuidedBackprop"
# supported arguments for dataset: "Cora", "DBLP"
```

3. Reproduce the plots

Our core results can be plotted using the following commands:

```
cd explanation_generation
generate_plots.py --task "node_removal" --dataset "Cora"
# supported arguments for task: "node_removal", "edge_removal", "edge_weights" (node_removal stands for node removal perturbations, edge_removal for edge removal perturbations, and edge_weights for edge weight perturbations)
# supported arguments for dataset: "Cora", "DBLP"
```

These commands plot the results for all of the models and explainers discussed in the paper. The paper also presents additional figures that support the discussion and analysis. To reproduce those plots, the Jupyter notebooks in the `/notebooks` folder can be run. All of the figures presented in the paper can also be found inside the `/img` folder.
