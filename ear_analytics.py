""" High level support for read and visualize
    information given by eacct command. """

import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


class Metric:
    """ Manage information of a metric. """

    def __init__(self, key, name=None, values=(0, 1)):
        self.key = key
        self.name = name
        self.range = values

    def set_name(self, name):
        """ Assigns the atribute 'name' of the instance. """
        self.name = name

    def set_values(self, values):
        """ Assigns the range atribute (a tuple of floats) of the instance. """
        self.range = values

    def norm_func(self, clip=True):
        """ Returns the normalization function. """
        return Normalize(vmin=self.range[0], vmax=self.range[1], clip=clip)


class Metrics:
    """ Manage all the metrics that this program can work with. """

    def __init__(self):
        self.metrics = dict()

    def add_metric(self, metric):
        """ Adds a new metric to the control data structure. """
        self.metrics[metric.key] = metric

    def get_metric(self, metric_key):
        """ Returns the metric corresponding to the key passed. """
        return self.metrics[metric_key]


def heatmap(n_sampl=32):
    """ Prepare the heatmap colormap, where 'n_sampl'
        defines the number of color samples. """

    vals = np.ones((n_sampl, 4))
    vals[:, 0] = np.linspace(1, 1, n_sampl)
    vals[:, 1] = np.linspace(1, 0, n_sampl)
    vals[:, 2] = np.linspace(0, 0, n_sampl)

    return ListedColormap(vals)



def main():

    metrics = Metrics()
    metrics.add_metric(Metric('cpi', 'CPI', (0.2, 3)))
    metrics.add_metric(Metric('avg_freq', 'AVG.FREQ', (1800000, 3600000)))
    metrics.add_metric(Metric('tpi', 'TPI', (0, 5)))
    metrics.add_metric(Metric('gbs', 'GBS', (0, 300)))
    metrics.add_metric(Metric('dc_node_pwr', 'DC-NODE-POWER', (100, 400)))
    metrics.add_metric(Metric('dram_pwr', 'DRAM-POWER', (5, 50)))
    metrics.add_metric(Metric('pck_pwr', 'PCK-POWER', (100, 400)))

    parser = argparse.ArgumentParser(description='Generate a heatmap graph'
                                                 ' showing the behaviour of '
                                                 'some metrics monitored with'
                                                 ' EARL.')

    parser.add_argument('input_file', help='Specifies the input file name to'
                                           ' read data from.')
    parser.add_argument('-m', '--metrics', action='append',
                        choices=list(metrics.metrics.keys()),
                        required=True)

    args = parser.parse_args()

    data_f = pd.read_csv(args.input_file, sep=';')
    group_by_node = data_f\
        .groupby(['NODENAME', 'ITERATIONS']).agg(lambda x: x).unstack(level=0)

    # Prepare x-axe range for iterations captured

    x_vals = group_by_node.index.values
    x_sampl = np.linspace(min(x_vals), max(x_vals), dtype=int)
    extent = [x_sampl[0]-(x_sampl[1]-x_sampl[0])//2,
              x_sampl[-1]+(x_sampl[1]-x_sampl[0])//2,
              0,
              1]

    # Compute the heatmap graph for each metric specified by the input

    for metric in args.metrics:
        metric_name = metrics.get_metric(metric).name

        m_data = group_by_node[metric_name].interpolate(limit_area='inside')
        m_data_array = m_data.values.transpose()

        # Create the resulting figure for current metric
        fig = plt.figure(figsize=[17.2, 9.6])
        fig.suptitle(metric_name)

        gs = GridSpec(nrows=len(m_data_array), ncols=2, width_ratios=(9.5, 0.5))
        # gs.tight_layout(fig)

        norm = metrics.get_metric(metric).norm_func()

        for i, _ in enumerate(m_data_array):
            ax = fig.add_subplot(gs[i, 0], ylabel=m_data.columns[i])
            ax.set_yticks([])
            data = np.array(m_data_array[i], ndmin=2)
            ax.imshow(data, cmap=heatmap(), norm=norm, aspect='auto', extent=extent)
            ax.set_xlim(extent[0], extent[1])
            if i != len(m_data_array) - 1:
                ax.set_xticklabels([])

        # fig.tight_layout()
        col_bar_ax = fig.add_subplot(gs[:, 1])
        fig.colorbar(cm.ScalarMappable(cmap=heatmap(), norm=norm), cax=col_bar_ax)
        plt.show()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
