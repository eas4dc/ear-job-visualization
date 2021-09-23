""" High level support for read and visualize
    information given by EARL. """

import argparse
import time

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

    # Convert time period from microseconds to seconds
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

    """
    # Plot, for each node, the average metrics requested for each process

    for idx in groupby_node_rank_mean.index.unique(level='NODE_ID'):
        fig = plt.figure(figsize=[6.4 * 3, 4.8 * 2])
        fig.suptitle(f"Node {idx}", size='xx-large', weight='bold')
        grid_s = GridSpec(nrows=2, ncols=3)

        x_data = np.array(groupby_node_rank_mean.loc[idx].index, ndmin=1)

        mpi = fig.add_subplot(grid_s[0, 0], ylabel='MPI calls',
                              title="Total MPI calls per process")
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['TOTAL MPI CALLS'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['TOTAL MPI CALLS'], ndmin=1)
        mpi.bar(x_data, y_data, yerr=y_err)

        perc_mpi = fig.add_subplot(grid_s[0, 1], ylabel='%MPI',
                                   title="% MPI per process")
        y_data = np.array(groupby_node_rank_mean.loc[idx]['PERC. MPI'],
                          ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['PERC. MPI'], ndmin=1)
        perc_mpi.bar(x_data, y_data, yerr=y_err)

        perc_sync_time = fig.add_subplot(grid_s[1, 0], ylabel='%Sync time',
                                         title="% Sync. time per process")
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['PERC. SYNC TIME'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['PERC. SYNC TIME'], ndmin=1)
        perc_sync_time.bar(x_data, y_data, yerr=y_err)

        perc_block_time = fig.add_subplot(grid_s[1, 1], ylabel='%Block time',
                                          title="% Block. time per process")
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['PERC. BLOCK TIME'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['PERC. BLOCK TIME'], ndmin=1)
        perc_block_time.bar(x_data, y_data, yerr=y_err)

        perc_collec_time = fig.add_subplot(grid_s[1, 2], ylabel='%Collec time',
                                           title="% Collec. time per process")
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['PERC. COLLEC TIME'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['PERC. COLLEC TIME'], ndmin=1)
        perc_collec_time.bar(x_data, y_data, yerr=y_err)

        cpi = fig.add_subplot(grid_s[0, 2], ylabel='CPI',
                              title="CPI per process")
        y_data = np.array(groupby_node_rank_mean.loc[idx]['CPI'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['CPI'], ndmin=1)
        cpi.bar(x_data, y_data, yerr=y_err)

        if not show:
            filename = f'resume_def_mpi_metrics_{idx}.jpg'
            plt.savefig(fname=filename, bbox_inches='tight')
        else:
            plt.show()
            plt.pause(0.001)

        fig = plt.figure(figsize=[6.4 * 3, 4.8 * 3])
        fig.suptitle(f"Node {idx}", size='xx-large', weight='bold')
        grid_s = GridSpec(nrows=3, ncols=3)

        sync_mpi_time_per_call = fig.add_subplot(grid_s[0, 0],
                                                 ylabel='Time per sync. call',
                                                 title='Average time per sync.'
                                                       ' call per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['TIME_PER_SYNC_CALL'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['TIME_PER_SYNC_CALL'], ndmin=1)
        sync_mpi_time_per_call.bar(x_data, y_data, yerr=y_err)

        block_mpi_time_per_call = fig.add_subplot(grid_s[0, 1],
                                                  ylabel='Time per block. cal'
                                                  'l', title='Average time per'
                                                  ' block. call per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['TIME_PER_BLOCK_CALL'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['TIME_PER_BLOCK_CALL'], ndmin=1)
        block_mpi_time_per_call.bar(x_data, y_data, yerr=y_err)

        collec_mpi_time_per_call = fig.add_subplot(grid_s[0, 2],
                                                   ylabel='Time per collec. ca'
                                                   'll', title='Average time p'
                                                   'er collec. call per proces'
                                                   's')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['TIME_PER_COLLEC_CALL'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['TIME_PER_COLLEC_CALL'], ndmin=1)
        collec_mpi_time_per_call.bar(x_data, y_data, yerr=y_err)

        total_sync_calls = fig.add_subplot(grid_s[1, 0],
                                           ylabel='Total sync. calls',
                                           title='Average sync. calls'
                                                 ' per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['SYNC_CALLS'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['SYNC_CALLS'], ndmin=1)
        total_sync_calls.bar(x_data, y_data, yerr=y_err)

        total_block_calls = fig.add_subplot(grid_s[1, 1],
                                            ylabel='Total block. calls',
                                            title='Average block. calls'
                                                  ' per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['BLOCK_CALLS'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['BLOCK_CALLS'], ndmin=1)
        total_block_calls.bar(x_data, y_data, yerr=y_err)

        total_collec_calls = fig.add_subplot(grid_s[1, 2],
                                             ylabel='Total collec. calls',
                                             title='Average collec. calls'
                                                   ' per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['COLLEC_CALLS'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['COLLEC_CALLS'], ndmin=1)
        total_collec_calls.bar(x_data, y_data, yerr=y_err)

        sync_mpi_time = fig.add_subplot(grid_s[2, 0],
                                        ylabel='Sync. time',
                                        title='Average sync. time'
                                              ' per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['SYNC_TIME'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['SYNC_TIME'], ndmin=1)
        sync_mpi_time.bar(x_data, y_data, yerr=y_err)

        block_mpi_time = fig.add_subplot(grid_s[2, 1],
                                         ylabel='Block. time',
                                         title='Average block. time'
                                               ' per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['BLOCK_TIME'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['BLOCK_TIME'], ndmin=1)
        block_mpi_time.bar(x_data, y_data, yerr=y_err)

        collec_mpi_time = fig.add_subplot(grid_s[2, 2],
                                          ylabel='Total collec. calls',
                                          title='Average collec. calls'
                                                ' per process')
        y_data = np.array(groupby_node_rank_mean
                          .loc[idx]['COLLEC_TIME'], ndmin=1)
        y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
            else np.array(groupby_node_rank_err
                          .loc[idx]['COLLEC_TIME'], ndmin=1)
        collec_mpi_time.bar(x_data, y_data, yerr=y_err)

        if not show:
            filename = f'resume_extra_mpi_metrics_{idx}.jpg'
            plt.savefig(fname=filename, bbox_inches='tight')
        else:
            plt.show()
            plt.pause(0.001)
    """

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

        # Plot, for each node, the average metrics requested for each process

        for node_id in groupby_node_rank_mean.index.unique(level='NODE_ID'):
            fig = plt.figure(figsize=[6.4, 4.8])

            fig.suptitle(build_title(f'{node_id} - {metric_name}', title),
                         size='xx-large', weight='bold')

            x_data = np.array(groupby_node_rank_mean
                              .loc[node_id].index, ndmin=1)

            mpi = fig.add_subplot(ylabel=f'{metric_name}',
                                  title=f'Average {metric_name} per process')
            y_data = np.array(groupby_node_rank_mean
                              .loc[node_id][metric_name], ndmin=1)
            y_err = np.zeros(len(y_data)) if groupby_node_rank_err is None \
                else np.array(groupby_node_rank_err
                              .loc[node_id][metric_name], ndmin=1)
            mpi.bar(x_data, y_data, yerr=y_err)

            if not show:
                filename = f'resume_{metric_name}_{node_id}.jpg'
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
            plt.savefig(fname=filename, bbox_inches='tight')
        else:
            plt.show()
            plt.pause(0.001)

        plt.close(fig)

        """
        for idx in m_data.columns.unique(level='NODE_ID'):
            m_data_array = m_data[idx].values.transpose()

            fig = plt.figure(figsize=[20.4, 0.5 * len(m_data[idx]
                                                      .columns) * 2])

            tit = f'Node {idx} per proc. {metric_name}'
            if title is not None:
                tit = f'{title}: node {idx} per proc. {metric_name}'
            fig.suptitle(tit, size='xx-large', weight='bold')

            grid_sp = GridSpec(nrows=len(m_data_array), ncols=2,
                               width_ratios=(9.5, 0.5))

            norm = Normalize(vmin=np.nanmin(m_data_array),
                             vmax=np.nanmax(m_data_array), clip=True)

            for i, _ in enumerate(m_data_array):
                axes = fig.add_subplot(grid_sp[i, 0],
                                       ylabel=m_data[idx].columns[i])
                axes.set_yticks([])
                data = np.array(m_data_array[i], ndmin=2)
                axes.imshow(data, cmap=heatmap(), norm=norm,
                            aspect='auto', extent=extent)
                axes.set_xlim(extent[0], extent[1])
                # Uncomment these lines to show timestamp labels on x axe
                # if i != len(m_data_array) - 1:
                #     axes.set_xticklabels([])
                axes.set_xticks([])

            col_bar_ax = fig.add_subplot(grid_sp[:, 1])
            fig.colorbar(cm.ScalarMappable(cmap=heatmap(), norm=norm),
                         cax=col_bar_ax)
            if show:
                plt.show()
                plt.pause(0.001)
            else:
                name = f'runtime_{metric_name}_{idx}.jpg'
                plt.savefig(fname=name, bbox_inches='tight')
        """


def runtime_parser_action_closure(metrics):
    """
    Closure function used to return the action
    function when `recursive` sub-command is called.
    """

    def run_parser_action(args):
        """ Action for `recursive` subcommand """
        print(args)
        runtime(args.input_file, metrics, args.metrics,
                args.show, args.title, args.output)

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
