""" High level support for read and visualize
    information given by eacct command. """

import argparse
import configparser

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

    def __str__(self):
        return f'{self.key}: {self.name} -- {self.range}'


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

    def __str__(self):
        res = ''
        for metric in self.metrics.values():
            res += (str(metric) + '\n')
        return res


def parse_float_tuple(in_str):
    """
    Given a tuple of two ints in str type, returns a tuple of two ints.
    """
    return tuple(float(k.strip()) for k in in_str[1:-1].split(','))


def read_ini(filename):
    """
    Load the configuration file `filename`
    """
    config = configparser.ConfigParser(converters={'tuple': parse_float_tuple})
    config.read(filename)
    return config


def init_metrics(config):
    """
    Based on configuration stored in `config`,
    inits the metric types used by this software.
    """
    mts = Metrics()

    for metric in config['METRICS']:
        mts.add_metric(Metric(metric, metric.upper(),
                       config['METRICS'].gettuple(metric)))

    return mts


def heatmap(n_sampl=32):
    """ Prepare the heatmap colormap, where 'n_sampl'
        defines the number of color samples. """

    vals = np.ones((n_sampl, 4))
    vals[:, 0] = np.linspace(1, 1, n_sampl)
    vals[:, 1] = np.linspace(1, 0, n_sampl)
    vals[:, 2] = np.linspace(0, 0, n_sampl)

    return ListedColormap(vals)


def resume(filename, base_freq, app_name=None, title=None):
    """ This function generates a graph of performance metrics given by
    `filename`.

    Performance metrics (Energy and Power save, and Time penalty)
    are ploted as percentage with respect to MONITORING (MO) results with the
    frequency `base_freq`.

    If the file `filename` contains resume information
    of multiple applications this function also accepts the parameter
    `app_name` which filters file's data to work only with `app_name`
    application results. """

    # TODO: control read file error
    data_f = pd.read_csv(filename)

    if app_name:
        # TODO: control app_name error
        data_f = data_f[data_f['APPLICATION'] == app_name]

    res_mns = data_f.groupby(['POLICY', 'DEF FREQ'])[['TIME(s)', 'ENERGY(J)',
                                                      'POWER(Watts)', 'CPI']].\
        mean()
    res_vs_base = res_mns
    ref = res_mns.loc['MO', base_freq]  # TODO: check base freq error

    res_vs_base['Time penalty (%)'] = ((res_vs_base['TIME(s)'] -
                                        ref['TIME(s)']) / ref['TIME(s)']) * 100
    res_vs_base['Energy save (%)'] = ((ref['ENERGY(J)'] -
                                       res_vs_base['ENERGY(J)']) /
                                      ref['ENERGY(J)']) * 100
    res_vs_base['Power save (%)'] = ((ref['POWER(Watts)'] -
                                      res_vs_base['POWER(Watts)']) /
                                     ref['POWER(Watts)']) * 100

    dropped = res_vs_base.drop(('MO', base_freq))

    results = dropped[['Time penalty (%)', 'Energy save (%)',
                       'Power save (%)']]

    tit = filename[:-4]
    if title:
        tit = title
    elif app_name:
        tit = app_name

    axes = results.plot(kind='bar', ylabel='(%)',
                        title=tit, figsize=(12.8, 9.6), rot=45)
    plt.grid(axis='y', ls='--', alpha=0.5)

    labels = np.ma.concatenate([results[serie].values
                                for serie in results.columns])
    rects = axes.patches

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2,
                  height + 0.1, '{:.2f}'.format(label),
                  ha='center', va='bottom')
        plt.show()


def recursive(filename, mtrcs, req_metrics, title=None):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `mtrcs` supported by ear_analytics.
    """

    data_f = pd.read_csv(filename, sep=';')
    group_by_node = data_f\
        .groupby(['NODENAME', 'ITERATIONS']).agg(lambda x: x).unstack(level=0)

    # Prepare x-axe range for iterations captured

    x_vals = group_by_node.index.values
    x_sampl = np.linspace(min(x_vals), max(x_vals), dtype=int)
    extent = [x_sampl[0]-(x_sampl[1]-x_sampl[0])//2,
              x_sampl[-1]+(x_sampl[1]-x_sampl[0])//2, 0, 1]

    # Compute the heatmap graph for each metric specified by the input

    for metric in req_metrics:
        metric_name = mtrcs.get_metric(metric).name

        m_data = group_by_node[metric_name].interpolate(limit_area='inside')
        m_data_array = m_data.values.transpose()

        # Create the resulting figure for current metric
        fig = plt.figure(figsize=[17.2, 9.6])

        tit = metric_name
        if title:
            tit = f'{title}: {metric_name}'
        fig.suptitle(tit)

        grid_sp = GridSpec(nrows=len(m_data_array), ncols=2,
                           width_ratios=(9.5, 0.5))

        norm = mtrcs.get_metric(metric).norm_func()

        for i, _ in enumerate(m_data_array):
            axes = fig.add_subplot(grid_sp[i, 0], ylabel=m_data.columns[i])
            axes.set_yticks([])
            data = np.array(m_data_array[i], ndmin=2)
            axes.imshow(data, cmap=heatmap(), norm=norm,
                        aspect='auto', extent=extent)
            axes.set_xlim(extent[0], extent[1])
            if i != len(m_data_array) - 1:
                axes.set_xticklabels([])

        # fig.tight_layout()
        col_bar_ax = fig.add_subplot(grid_sp[:, 1])
        fig.colorbar(cm.ScalarMappable(cmap=heatmap(), norm=norm),
                     cax=col_bar_ax)
        plt.show()
        plt.pause(0.001)


def res_parser_action(args):
    """ Action for `resume` subcommand """
    resume(args.input_file, args.base_freq, args.app_name, args.title)


def rec_parser_action(args):
    """ Action for `recursive` subcommand """
    recursive(args.input_file, metrics, args.metrics, args.title)


def main():
    """ Entry method. """

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='ear_analytics',
                                     description='High level support for read '
                                     'and visualize information files given by'
                                     ' eacct command.')
    parser.add_argument('input_file', help='Specifies the input file name to'
                        ' read data from.', type=str)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save', action='store_true',
                       help='Activate the flag to store resulting figures'
                       ' (default).')
    group.add_argument('--show', action='store_true',
                       help='Show the resulting figure.')

    parser.add_argument('-o', '--output',
                        help='Sets the output image name.'
                        ' Only valid if `--save` flag is set (default).')
    parser.add_argument('-t', '--title',
                        help='Set the resulting figure title.')

    subparsers = parser.add_subparsers(help='The two functionalities currently'
                                       ' supported by this program.',
                                       description='Type `ear_analytics` '
                                       '<dummy_filename> {recursive,resume} -h'
                                       ' to get more info of each subcommand')

    # create the parser for the `recursive` command
    parser_rec = subparsers.add_parser('recursive',
                                       help='Generate a heatmap graph showing'
                                       ' the behaviour of some metrics monitor'
                                       'ed with EARL. The file must be outpute'
                                       'd by `eacct -r` command.')
    parser_rec.add_argument('-m', '--metrics', action='append',
                            choices=list(metrics.metrics.keys()),
                            required=True, help='Specify which metrics you wan'
                            't to visualize.')
    parser_rec.set_defaults(func=rec_parser_action)

    # create the parser for the `resume` command
    parser_res = subparsers.add_parser('resume',
                                       help='Generate a resume about Energy'
                                       ' and Power save, and Time penalty of'
                                       ' an application monitored with EARL.')
    parser_res.add_argument('base_freq', help='Specify which'
                            ' frequency is used as base '
                            'reference for computing and showing'
                            ' savings and penalties in the figure.')
    parser_res.add_argument('--app_name', help='Set the application name to'
                            ' get resume info.')
    parser_res.set_defaults(func=res_parser_action)

    args = parser.parse_args()
    args.func(args)


metrics = init_metrics(read_ini('config.ini'))

if __name__ == '__main__':
    main()
