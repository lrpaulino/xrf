import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def rpiq(y_true, y_pred):
    """
    Calcula o RPIQ (Ratio of Performance to Interquartile Distance).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calcula o intervalo interquartil dos valores reais
    q3, q1 = np.percentile(y_true, [75, 25])
    iq = q3 - q1

    # Calcula o RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calcula RPIQ (evita divisão por zero)
    return iq / rmse if rmse != 0 else np.inf


def plot_metrics(df_scores, target):
    sns.set_palette('deep')
    ax = sns.catplot(
        data=df_scores,
        x='model', y='value', col='metric',
        hue='model',
        kind='bar',
        sharey=False,
        height=5,
        aspect=0.8,
    )

    axes_list = list(ax.axes.flat)
    model_names = list(df_scores['model'].unique())
    metric_names = list(df_scores['metric'].unique())

    def axis_data_limits(ax_obj):
        mins, maxs = [], []
        for cont in ax_obj.containers:
            for patch in cont:
                y0 = patch.get_y()
                y1 = y0 + patch.get_height()
                mins.append(y0)
                maxs.append(y1)
        if mins and maxs:
            return min(mins), max(maxs)
        else:
            return ax_obj.get_ylim()

    def base_metric_name(metric):
        return str(metric).split('[')[0].strip().lower()

    facets_limits = [axis_data_limits(a) for a in axes_list]
    base_names = [base_metric_name(m) for m in metric_names]

    groups = {}
    for i, base in enumerate(base_names):
        groups.setdefault(base, []).append(i)

    def padded_range(indices):
        ymin = min(0.0, min(facets_limits[k][0] for k in indices))
        ymax = max(facets_limits[k][1] for k in indices)
        if ymax == ymin:
            pad = 0.1 if ymax == 0 else abs(ymax) * 0.1
        else:
            pad = (ymax - ymin) * 0.1
        return ymin, ymax + pad

    for i, axes in enumerate(axes_list):
        base = base_names[i]
        if base in ('r2', 'r²'):
            axes.set_ylim(0, 1.0)
        elif base == 'rmse':
            ymin, ymax = padded_range(groups['rmse'])
            axes.set_ylim(ymin, ymax)
        else:
            ymin, ymax = padded_range([i])
            axes.set_ylim(ymin, ymax)

    # Improve styling
    ax.set_titles('{col_name}', size=14, weight='bold')
    ax.set(xlabel=None, ylabel='Score')
    ax.despine(left=True)

    # Add value labels with mean ± std formatting
    std_table = df_scores.groupby(['model', 'metric'])['value'].std()

    for i, axes in enumerate(axes_list):
        for j, container in enumerate(axes.containers):
            bar_value = container[0].get_height()
            std_value = std_table.loc[model_names[j], metric_names[i]]

            labels = [f'{bar_value:.2f} ± {std_value:.2f}']

            axes.bar_label(container, labels=labels, padding=15, fontsize=9)
            axes.grid(axis='y', alpha=0.3, linestyle='--')
            axes.set_axisbelow(True)

    plt.suptitle(f'{target}', size=25)
    plt.subplots_adjust(top=0.85)
    plt.show()
    plt.close()
