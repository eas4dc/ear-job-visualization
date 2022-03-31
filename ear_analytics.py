""" High level support for read and visualize
    information given by EARL. """

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
import colorcet as cc

from common.io_api import read_data, read_ini, configure_verbosity, VERBOSE
from common.metrics import init_metrics
from common.utils import filter_by_job_step_app


def resume(filename, base_freq, app_id=None, job_id=None,
           show=False, output=None, title=None):
    """
    This function generates a graph of performance metrics given by `filename`.

    Performance metrics (Energy and Power save, and Time penalty)
    are ploted as percentage with respect to MONITORING (MO) results with the
    frequency `base_freq`.

    If the file `filename` contains resume information
    of multiple applications this function also accepts the parameter
    `app_name` and/or `job_id` which filters file's data to work only with'
    ' `app_name` and/or `job_id` application results.
    """

    def preprocess_df(data_f):
        """
        Pre-process DataFrame `data_f` to get workable data.
        """
        return (data_f
                .assign(
                  def_freq=lambda x: round(data_f['DEF.FREQ'] * 10**-6, 4),
                  avg_cpu_freq=lambda x:
                  round(data_f['AVG.CPUFREQ'] * 10**-6, 4),
                  avg_imc_freq=lambda x:
                  round(data_f['AVG.IMCFREQ'] * 10**-6, 4),
                  energy=lambda x:
                  data_f['TIME'] * data_f['DC-NODE-POWER'],
                )
                .drop(['DEF.FREQ', 'AVG.CPUFREQ', 'AVG.IMCFREQ'], axis=1)
                )

    # Filter rows and pre-process data
    data_f = preprocess_df(filter_by_job_step_app(read_data(filename),
                           job_id=job_id, app_id=app_id))

    # Compute average step energy consumed
    energy_sums = (data_f
                   .groupby(['POLICY', 'def_freq', 'STEP_ID'])['energy']
                   .sum()
                   )

    # resume data
    re_dat = (pd
              .concat([data_f
                      .groupby(['POLICY', 'def_freq', 'STEP_ID'])
                      [['TIME', 'DC-NODE-POWER', 'avg_cpu_freq',
                        'avg_imc_freq']]
                      .mean(),
                      energy_sums],
                      axis=1
                      )
              .groupby(['POLICY', 'def_freq'])
              .mean()
              )

    # reference data
    ref_data = re_dat.loc['monitoring', base_freq]

    # Computes savings and penalties
    re_dat['Time penalty'] = (
            (re_dat['TIME'] - ref_data['TIME'])
            / ref_data['TIME']
            ) * 100
    re_dat['Energy save'] = (
            (ref_data['energy'] - re_dat['energy'])
            / ref_data['energy']
            ) * 100
    re_dat['Power save'] = (
            (ref_data['DC-NODE-POWER'] - re_dat['DC-NODE-POWER'])
            / ref_data['DC-NODE-POWER']
            ) * 100

    dropped = re_dat.drop(('monitoring', base_freq))

    results = dropped[['Time penalty', 'Energy save',
                       'Power save']]

    # Get avg. cpu and imc frequencies
    freqs = dropped[['avg_cpu_freq', 'avg_imc_freq']]

    # Prepare and create the plot
    tit = 'resume'
    if title:
        tit = title
    elif app_id:
        tit = app_id + f' vs. {base_freq} GHz'

    axes = results.plot(kind='bar', figsize=(12.8, 9.6),
                        rot=45, legend=False, fontsize=20)
    axes.set_xlabel('POLICY, def. Freq (GHz)', fontsize=20)

    plt.gcf().suptitle(tit, fontsize='22', weight='bold')

    ax2 = axes.twinx()
    freqs.plot(ax=ax2,  ylim=(0, 3.5), color=['cyan', 'purple'],
               linestyle='-.', legend=False, fontsize=20)
    ax2.set_ylabel(ylabel='avg. Freq (GHz)', labelpad=20.0, fontsize=20)

    # create the legend
    handles_1, labels_1 = axes.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()

    axes.legend(handles_1 + handles_2, labels_1 + labels_2, loc=0, fontsize=15)

    # Plot a grid
    plt.grid(axis='y', ls='--', alpha=0.5)

    # Plot value labels above the bars
    labels = np.ma.concatenate([results[serie].values
                                for serie in results.columns])
    rects = axes.patches

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2,
                  height + 0.1, '{:.2f}%'.format(label),
                  ha='center', va='bottom', fontsize=12)
    if show:
        plt.show()
    else:
        name = 'resume.jpg'
        if output is not None:
            name = output
        plt.savefig(fname=name, bbox_inches='tight')


def heatmap(n_sampl=32):
    """ Prepare the heatmap colormap, where 'n_sampl'
        defines the number of color samples. """

    vals = np.ones((n_sampl, 4))
    vals[:, 0] = np.linspace(1, 1, n_sampl)
    vals[:, 1] = np.linspace(1, 0, n_sampl)
    vals[:, 2] = np.linspace(0, 0, n_sampl)

    return ListedColormap(vals)


def runtime(filename, mtrcs, req_metrics,
            show=False, title=None, job_id=None, step_id=None, output=None):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `mtrcs` supported by ear_analytics.
    """

    group_by_node = (filter_by_job_step_app(read_data(filename), job_id=job_id,
                                            step_id=step_id)
                     .groupby(['NODENAME', 'TIMESTAMP'])
                     .agg(lambda x: x).unstack(level=0)
                     )
    # Prepare x-axe range for iterations captured
    x_sampl = np.linspace(min(group_by_node.index.values),
                          max(group_by_node.index.values), dtype=int)
    extent = [x_sampl[0]-(x_sampl[1]-x_sampl[0])//2,
              x_sampl[-1]+(x_sampl[1]-x_sampl[0])//2, 0, 1]

    # Compute the heatmap graph for each metric specified by the input
    for metric in req_metrics:
        metric_name = mtrcs.get_metric(metric).name

        m_data = group_by_node[metric_name]  # .interpolate(method='bfill',

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

        # Create the resulting figure for current metric
        fig = plt.figure(figsize=[20.4, 0.5 * len(m_data.columns) * 2])

        tit = metric_name
        if title is not None:
            tit = f'{title}: {metric_name}'
        fig.suptitle(tit, y=0.93, size=22, weight='bold')

        grid_sp = GridSpec(nrows=len(m_data_array), ncols=2,
                           width_ratios=(9.5, 0.5))

        # Switch below lines in order to apply the specific metric range
        # norm = mtrcs.get_metric(metric).norm_func()
        norm = Normalize(vmin=np.nanmin(m_data_array),
                         vmax=np.nanmax(m_data_array), clip=True)

        for i, _ in enumerate(m_data_array):
            axes = fig.add_subplot(grid_sp[i, 0], ylabel=m_data.columns[i])
            axes.set_yticks([])
            data = np.array(m_data_array[i], ndmin=2)
            axes.imshow(data, cmap=cc.cm.bmy, norm=norm,
                        aspect='auto', extent=extent)
            axes.set_xlim(extent[0], extent[1])
            # Uncomment these lines to show timestamp labels on x axe
            if i != len(m_data_array) - 1:
                axes.set_xticklabels([])
            # axes.set_xticks([])

        # fig.tight_layout()
        col_bar_ax = fig.add_subplot(grid_sp[:, 1])
        fig.colorbar(cm.ScalarMappable(cmap=cc.cm.bmy, norm=norm),
                     cax=col_bar_ax)
        if show:
            plt.show()
            plt.pause(0.001)
        else:
            name = f'runtime_{metric_name}.jpg'
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
            plt.savefig(fname=name, bbox_inches='tight')


def runtime_parser_action_closure(metrics):
    """
    Closure function used to return the action
    function when `recursive` sub-command is called.
    """

    def run_parser_action(args):
        """ Action for `recursive` subcommand """
        configure_verbosity(args.verbosity)
        # VERBOSE(3, f'[INFO]Arguments parsed: {args}')
        print(args)
        runtime(args.input_file, metrics, args.metrics,
                args.show, args.title, args.jobid, args.stepid, args.output)

    return run_parser_action


def res_parser_action(args):
    """ Action for `resume` subcommand """
    configure_verbosity(args.verbosity)
    # VERBOSE(3, f'[INFO]Arguments parsed: {args}')
    resume(args.input_file, args.base_freq, args.app_name,
           args.jobid, args.show, args.output, args.title)


def build_parser(metrics):
    """
    Given the used `metrics`,
    returns a parser to read and check command line arguments.
    """
    class CustomHelpFormatter(argparse.HelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=40, width=80)

        def _format_action_invocation(self, action):
            if not action.option_strings or action.nargs == 0:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings) + ' ' + args_string

    def formatter(prog):
        return CustomHelpFormatter(prog)

    parser = argparse.ArgumentParser(prog='ear_analytics',
                                     description='High level support for read '
                                     'and visualize information files given by'
                                     ' EARL.', formatter_class=formatter)
    parser.add_argument('--version', action='version', version='%(prog)s 2.1')
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
    parser.add_argument('-j', '--jobid', type=int,
                        help='Sets the JOB ID you are working'
                        ' with.')
    parser.add_argument('-v', '--verbosity', action="count",
                        help="increase output verbosity"
                        "(e.g., -vv is more than -v)")

    subparsers = parser.add_subparsers(help='The two functionalities currently'
                                       ' supported by this program.',
                                       description='Type `ear_analytics '
                                       '<dummy_filename> {runtime,resume} -h'
                                       '` to get more info of each subcommand')

    # create the parser for the `recursive` command
    parser_run = subparsers.add_parser('runtime',
                                       help='Generate a heatmap graph showing'
                                       ' the behaviour of some metrics monitor'
                                       'ed with EARL. The file must be outpute'
                                       "d by `eacct -r` command or by an EAR's"
                                       ' report plugin.',
                                       formatter_class=formatter)
    parser_run.add_argument('-s', '--stepid', type=int,
                            help='Sets the STEP ID of the job you are working'
                            ' with.')
    parser_run.add_argument('-m', '--metrics', nargs='+',
                            choices=list(metrics.metrics.keys()),
                            required=True, help='Specify which metrics you wan'
                            't to visualize.')
    parser_run.set_defaults(func=runtime_parser_action_closure(metrics))

    # create the parser for the `resume` command
    parser_res = subparsers.add_parser('resume',
                                       help='Generate a resume about Energy'
                                       ' and Power save, and Time penalty of'
                                       ' an application monitored with EARL.',
                                       formatter_class=formatter)
    parser_res.add_argument('base_freq', help='Specify which'
                            ' frequency is used as base '
                            'reference for computing and showing'
                            ' savings and penalties in the figure.',
                            type=float)
    parser_res.add_argument('--app_name', help='Set the application name to'
                            ' get resume info.')
    parser_res.set_defaults(func=res_parser_action)

    return parser


def main():
    """ Entry method. """

    # Read configuration file and init `metrics` data structure
    metrics = init_metrics(read_ini('config.ini'))

    # create the top-level parser
    parser = build_parser(metrics)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
