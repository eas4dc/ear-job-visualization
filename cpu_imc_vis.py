""" Generate a 2D heatmap based on CPU and IMC freqs. """
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io_api import read_ini, read_data
from metrics import init_metrics
from utils import filter_by_job_step_app


def make_heatmap(filename, mtrcs, req_metrics, show=False,
                 title=None, job_id=None):
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
                           job_id=job_id))

    grouped_by_step = data_f.groupby('STEPID').mean()

    grouped_by_cpu_imc = grouped_by_step.groupby(['def_freq', 'avg_imc_freq'])\
        .mean().rename(index=lambda x: round(x, 4))

    def_freq_vals = grouped_by_cpu_imc.index.unique(level='def_freq')
    avg_imc_vals = grouped_by_cpu_imc.index.unique(level='avg_imc_freq')

    for metric in req_metrics:
        metric_name = mtrcs.get_metric(metric).name

        heatmap = pd.DataFrame(index=def_freq_vals, columns=avg_imc_vals)
        idxs = [(x, y) for x in heatmap.index for y in heatmap.columns]

        for idx in idxs:
            heatmap.loc[idx[0]][idx[1]] = \
                grouped_by_cpu_imc.loc[idx[0], idx[1]][metric_name]

        heatmap.index.name = None
        heatmap.columns.name = None
        heatmap.apply(pd.to_numeric).style.background_gradient(axis=None)

        fig = plt.figure()
        tit = metric_name
        if title is not None:
            tit = title + ': ' + metric_name
        fig.suptitle(tit, size='xx-large', weight='bold')

        axes = sns.heatmap(heatmap.astype(float), cmap='YlOrRd')
        fig.add_axes(axes)
        plt.savefig(fname=f'heatmap_{metric_name}.jpg', bbox_inches='tight')


def parser_action_closure(metrics):
    """
    Closure function used to return the action
    function when `recursive` sub-command is called.
    """

    def hm_parser_action(args):
        """ Action for `recursive` subcommand """
        print(args)
        make_heatmap(args.input_file, metrics, args.metrics,
                     args.show, args.title, args.jobid)

    return hm_parser_action


def build_parser(metrics):
    """
    Given the used `metrics`,
    returns a parser to read and check command line arguments.
    """
    parser = argparse.ArgumentParser(prog='ear_cpu_imc_corr',
                                     description='High level support for read '
                                     'and visualize the correlation of metrics'
                                     ' given a fixed CPU and IMC freqs.')
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
                        help='Set resulting figures "title + %metric".')
    parser.add_argument('-m', '--metrics', nargs='+',
                        choices=list(metrics.metrics.keys()),
                        required=True, help='Specify which metrics you wan'
                        't to visualize.')

    parser.set_defaults(func=parser_action_closure(metrics))

    return parser


def main():
    """ Entry method. """

    # Read configuration file and init `metrics` data structure
    metrics = init_metrics(read_ini('config.ini'))
    # print(f'METRICS CONFIG\n{metrics.__str__()}')

    # create the top-level parser
    parser = build_parser(metrics)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
