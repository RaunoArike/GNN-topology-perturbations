import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def plot_with_confidence_intervals(avg_freq, results, label, color):
    mean_values = np.mean(results, axis=1)
    sem_values = sem(results, axis=1)  # SEM is standard error of the mean

    # Calculate the confidence intervals (95% confidence level)
    ci_lower = mean_values - 1.96 * sem_values
    ci_upper = mean_values + 1.96 * sem_values

    plt.plot(avg_freq, mean_values, label=label, color=color, linewidth=2, marker='o')

    # Plot the confidence intervals as shaded areas
    plt.fill_between(avg_freq, ci_lower, ci_upper, color=color, alpha=0.2)


def plot_baseline(baseline, label, color):
    mean = np.mean(baseline)
    sem_baseline = sem(baseline)

    ci_lower = mean - 1.96 * sem_baseline
    ci_upper = mean + 1.96 * sem_baseline

    plt.axhline(y=mean, color=color, linestyle='--', label=label)
    plt.fill_between([0, 1], ci_lower, ci_upper, color=color, alpha=0.2, transform=plt.gca().get_yaxis_transform())
    