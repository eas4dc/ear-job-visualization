""" High level support for read and visualize
    information given by EARL. """

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import colorcet as cc

from io_api import read_data, read_ini
from metrics import init_metrics
from utils import filter_by_job_step_app


def heatmap(n_sampl=32):
    """ Prepare the heatmap colormap, where 'n_sampl'
        defines the number of color samples. """

    vals = np.ones((n_sampl, 4))
    vals[:, 0] = np.linspace(1, 1, n_sampl)
    vals[:, 1] = np.linspace(1, 0, n_sampl)
    vals[:, 2] = np.linspace(0, 0, n_sampl)

    return ListedColormap(vals)


def runtime(filename, mtrcs, req_metrics,
            show=False, title=None, output=None, err=False):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `mtrcs` supported by ear_analytics.
    """

    def build_title(def_title, title):
        if title is not None:
            return f'{title}: {def_title}'
        return def_title

    # Read data
    data_f = filter_by_job_step_app(read_data(filename))

    # Pre-process and create new Series derived from existing ones

    # Convert time period from microseconds to miliseconds
    data_f['TIME PERIOD'] = round(data_f['TIME PERIOD'] * (10**-3), 2)

    # Total synchronization calls
    data_f['SYNC_CALLS'] = (data_f['PERC.SYNC_CALLS'] / 100) \
        * data_f['TOTAL_MPI_CALLS']

    # Synchronization time
    data_f['SYNC_TIME'] = (data_f['PERC.SYNC_TIME'] / 100) \
        * data_f['TIME PERIOD']

    # Time per synchronization call
    data_f['TIME_PER_SYNC_CALL'] = (data_f['SYNC_TIME']
                                    / data_f['SYNC_CALLS'])

    data_f['BLOCK_CALLS'] = (data_f['PERC.BLOCK_CALLS'] / 100) \
        * data_f['TOTAL_MPI_CALLS']

    data_f['BLOCK_CALLS_SEC'] = (data_f['BLOCK_CALLS'] /
                                 round(data_f['TIME PERIOD']
                                       * (10**-3), 2))

    data_f['BLOCK_TIME'] = (data_f['PERC.BLOCK_TIME'] / 100) \
        * data_f['TIME PERIOD']

    data_f['TIME_PER_BLOCK_CALL'] = (data_f['BLOCK_TIME']
                                     / data_f['BLOCK_CALLS'])

    data_f['COLLEC_CALLS'] = (data_f['PERC.COLLEC_CALLS'] / 100) \
        * data_f['TOTAL_MPI_CALLS']

    data_f['COLLEC_TIME'] = (data_f['PERC.COLLEC_TIME'] / 100) \
        * data_f['TIME PERIOD']

    data_f['TIME_PER_COLLEC_CALL'] = (data_f['COLLEC_TIME']
                                      / data_f['COLLEC_CALLS'])

    # Grouping info

    groupby_node_rank_mean = (data_f
                              .groupby(['NODE_ID', 'GLOBAL RANK'])
                              .mean()
                              )

    groupby_node_rank_err = None if not err else (data_f
                                                  .groupby(['NODE_ID',
                                                            'GLOBAL RANK'])
                                                  .agg(np.std)
                                                  )

    # Group data of any NODENAME across all timestamps
    # groupby_lrank_ts = data_f.groupby(['NODE_ID', 'LOCAL RANK', 'TIMESTAMP'])
    #    .agg(lambda x: x).unstack(level=0).unstack(level=0)
    groupby_lrank_ts = (data_f.groupby(['GLOBAL RANK', 'TIMESTAMP'])
                        .agg(lambda x: x).unstack(level=0)
                        )

    # Prepare x-axe range for iterations captured
    x_sampl = np.linspace(min(groupby_lrank_ts.index.values),
                          max(groupby_lrank_ts.index.values), dtype=int)
    extent = [x_sampl[0]-(x_sampl[1]-x_sampl[0])//2,
              x_sampl[-1]+(x_sampl[1]-x_sampl[0])//2, 0, 1]

    # Compute the heatmap graph for each metric specified by the input

    for metric in req_metrics:
        metric_name = mtrcs.get_metric(metric).name

        # Get the maximum values for y-axes
        max_y_lim_mean = groupby_node_rank_mean[metric_name].max()
        max_y_lim_err = 0 if groupby_node_rank_err is None \
            else groupby_node_rank_err[metric_name].max()

        # Plot, for each node, the average metrics requested for each process

        for node_id in groupby_node_rank_mean.index.unique(level='NODE_ID'):
            fig = plt.figure(figsize=[6.4, 4.8])

            fig.suptitle(build_title(f'{node_id} - {metric_name}', title),
                         size='xx-large', weight='bold')

            x_data = np.array(groupby_node_rank_mean
                              .loc[node_id].index, ndmin=1)

            axs = fig.add_subplot(ylabel=f'{metric_name}',
                                  title=f'Average {metric_name} per process')
            y_data = np.array(groupby_node_rank_mean
                              .loc[node_id][metric_name], ndmin=1)
            y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
                else np.array(groupby_node_rank_err
                              .loc[node_id][metric_name], ndmin=1)
            axs.set_ylim(top=max_y_lim_mean + max_y_lim_err)
            axs.bar(x_data, y_data, yerr=y_err)

            if not show:
                filename = f'resume_{metric_name}_{node_id}.jpg'
                if output is not None:
                    if os.path.isdir(output):
                        print(f'storing file {filename} to {output} directory')
                    else:
                        print(f'{output} directory does not exist! creating'
                              f' directory and storing {filename} inside it.')
                        os.makedirs(output)
                    if output[-1] == '/':
                        filename = output + filename
                    else:
                        filename = output + '/' + filename
                print(f'Saving figure to {filename}')
                plt.savefig(fname=filename, bbox_inches='tight')
            else:
                plt.show()
                plt.pause(0.001)

            plt.close(fig)

        m_data = groupby_lrank_ts[metric_name]

        for key in x_sampl:
            if key not in m_data.index:
                m_data = m_data.append(pd.Series(name=key, dtype=object),
                                       verify_integrity=True)
                m_data.sort_index(inplace=True)

        m_data = m_data.interpolate(method='bfill', limit_area='inside')

        for idx in m_data.index.values:
            if idx not in x_sampl:
                m_data.drop(idx, inplace=True)

        m_data_array = m_data.values.transpose()

        norm = Normalize(vmin=np.nanmin(m_data_array),
                         vmax=np.nanmax(m_data_array), clip=True)

        fig = plt.figure(figsize=[20.4, 0.5 * len(m_data.columns) * 2])

        fig.suptitle(build_title(f'Per proc. {metric_name}', title),
                     size='xx-large', weight='bold')

        grid_s = GridSpec(nrows=len(m_data_array), ncols=2,
                          width_ratios=(9.5, 0.5))

        for i, _ in enumerate(m_data_array):
            axes = fig.add_subplot(grid_s[i, 0], ylabel=m_data.columns[i])
            axes.set_yticks([])
            data = np.array(m_data_array[i], ndmin=2)

            axes.imshow(data, cmap=cc.cm.bmy, norm=norm,
                        aspect=5.0, extent=extent)
            axes.set_xlim(extent[0], extent[1])
            axes.set_xticks([])

        fig.tight_layout()
        col_bar_ax = fig.add_subplot(grid_s[:, 1])
        fig.colorbar(cm.ScalarMappable(cmap=cc.cm.bmy, norm=norm),
                     cax=col_bar_ax)

        if not show:
            filename = f'runtime_{metric_name}.jpg'
            if output is not None:
                if os.path.isdir(output):
                    print(f'storing file {filename} to {output} directory')
                else:
                    print(f'{output} directory does not exist! creating'
                          f' directory and storing {filename} inside it.')
                    os.makedirs(output)
                if output[-1] == '/':
                    filename = output + filename
                else:
                    filename = output + '/' + filename
            print(f'Saving figure to {filename}')
            plt.savefig(fname=filename, bbox_inches='tight')
        else:
            plt.show()
            plt.pause(0.001)

        plt.close(fig)


def runtime_parser_action_closure(metrics):
    """
    Closure function used to return the action
    function when `recursive` sub-command is called.
    """

    def run_parser_action(args):
        """ Action for `recursive` subcommand """
        print(args)
        runtime(args.input_file, metrics, args.metrics,
                args.show, args.title, args.output, args.error)

    return run_parser_action


def build_parser(metrics):
    """
    Given the used `metrics`,
    returns a parser to read and check command line arguments.
    """
    parser = argparse.ArgumentParser(prog='per-proc-ear_analytics',
                                     description='High level support for read '
                                     'and visualize per process information fi'
                                     'les given by EARL.')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('input_file', help='Specifies the input file(s) name(s'
                        ') to read data from.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save', action='store_true',
                       help='Activate the flag to store resulting figures'
                       ' (default).')
    group.add_argument('--show', action='store_true',
                       help='Show the resulting figure.')

    parser.add_argument('-t', '--title',
                        help='Set the resulting figure title.')
    parser.add_argument('-o', '--output',
                        help='Sets the output image name.'
                        ' Only valid if `--save` flag is set.')
    parser.add_argument('-e', '--error', action='store_true',
                        help='Show error bars based on standard'
                             ' deviation for bar plots.')

    parser.add_argument('-m', '--metrics', nargs='+',
                        choices=list(metrics.metrics.keys()),
                        required=True, help='Specify which metrics you wan'
                        't to visualize.')

    return parser


def main():
    """ Entry method. """

    # Read configuration file and init `metrics` data structure
    metrics = init_metrics(read_ini('config.ini'))
    # print(f'METRICS CONFIG\n{metrics.__str__()}')

    action = runtime_parser_action_closure(metrics)

    # create the top-level parser
    parser = build_parser(metrics)

    args = parser.parse_args()

    action(args)


if __name__ == '__main__':
    main()
