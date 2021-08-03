""" High level support for read and visualize
    information given by EARL. """

import argparse
import configparser
import os
from itertools import dropwhile

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


def filter_by_job_step_app(data_f, job_id=None, step_id=None, app_id=None):
    """
    Filters the DataFrame `data_f` by `job_id`
    and/or `step_id` and/or `app_id`.
    """

    def mask(data_f, key, value):
        if value is not None:
            return data_f[data_f[key] == value]
        return data_f

    pd.DataFrame.mask = mask

    return (data_f
            .mask('APP_ID', app_id)
            .mask('JOB_ID', job_id)
            .mask('STEPID', step_id)
            )


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
    ' `app_name` and/or `job_id` application results. """

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
                  ENERGY_TAG=lambda x:
                  data_f['TIME'] * data_f['DC-NODE-POWER'],
                )
                .drop(['DEF.FREQ', 'AVG.CPUFREQ', 'AVG.IMCFREQ'], axis=1)
                )

    # Filter rows and pre-process data
    data_f = preprocess_df(filter_by_job_step_app(read_data(filename),
                           job_id=job_id, app_id=app_id))

    # Compute average step energy consumed
    energy_sums = (data_f
                   .groupby(['POLICY', 'def_freq', 'STEP_ID'])['ENERGY_TAG']
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
    re_dat['Time penalty (%)'] = (
            (re_dat['TIME'] - ref_data['TIME'])
            / ref_data['TIME']
            ) * 100
    re_dat['Energy save (%)'] = (
            (ref_data['ENERGY_TAG'] - re_dat['ENERGY_TAG'])
            / ref_data['ENERGY_TAG']
            ) * 100
    re_dat['Power save (%)'] = (
            (ref_data['DC-NODE-POWER'] - re_dat['DC-NODE-POWER'])
            / ref_data['DC-NODE-POWER']
            ) * 100

    dropped = re_dat.drop(('monitoring', base_freq))

    results = dropped[['Time penalty (%)', 'Energy save (%)',
                       'Power save (%)']]

    # Get avg. cpu and imc frequencies
    freqs = dropped[['avg_cpu_freq', 'avg_imc_freq']]

    # Prepare and create the plot
    tit = 'resume'
    if title:
        tit = title
    elif app_id:
        tit = app_id + f' vs. {base_freq} GHz'

    axes = results.plot(kind='bar', ylabel='(%)', title=tit,
                        figsize=(12.8, 9.6), rot=45, legend=False)
    ax2 = axes.twinx()
    freqs.plot(ax=ax2, ylabel='avg. Freq (GHz)', ylim=(0, 3.5),
               color=['cyan', 'purple'], linestyle='-.', legend=False)

    # create the legend
    handles_1, labels_1 = axes.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()

    axes.legend(handles_1 + handles_2, labels_1 + labels_2, loc=0)

    # Plot a grid
    plt.grid(axis='y', ls='--', alpha=0.5)

    # Plot value labels above the bars
    labels = np.ma.concatenate([results[serie].values
                                for serie in results.columns])
    rects = axes.patches

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2,
                  height + 0.1, '{:.2f}'.format(label),
                  ha='center', va='bottom')
    if show:
        plt.show()
    else:
        name = 'resume.jpg'
        if output is not None:
            name = output
        plt.savefig(fname=name, bbox_inches='tight')


def read_data(file_path, sep=';'):
    """
    This function reads data properly whether the input `file_path` is a list
    of concret files, is a directory #,or a list of directories,# and returns a
    pandas.DataFrame object.
    """

    def load_files(filenames, base_path=None, sep=sep):
        for filename in filenames:
            if base_path:
                path_file = base_path + '/' + filename
            else:
                path_file = filename
            yield pd.read_csv(path_file, sep=sep)

    if isinstance(file_path, str):
        # `file_path` is only a string containing some file or a directory
        if os.path.isfile(file_path):
            print(f'reading file {file_path}')
            data_f = pd.concat(load_files([file_path]), ignore_index=True)
        elif os.path.isdir(file_path):
            print(f'reading files contained in directory {file_path}')
            print(f'{os.listdir(file_path)}')
            data_f = pd.concat(load_files(os.listdir(file_path),
                               base_path=file_path),
                               ignore_index=True)
        else:
            print(f'{file_path} does not exist!')
            raise FileNotFoundError
    elif isinstance(file_path, list):
        # `file_path` is a list containing files
        try:
            no_exists_file = next(dropwhile(os.path.exists, file_path))
            print(f'file {no_exists_file} does not exist!')
            raise FileNotFoundError
        except StopIteration:
            print(f'reading files {file_path}')
            data_f = pd.concat(load_files(file_path), ignore_index=True)
    return data_f


def heatmap(n_sampl=32):
    """ Prepare the heatmap colormap, where 'n_sampl'
        defines the number of color samples. """

    vals = np.ones((n_sampl, 4))
    vals[:, 0] = np.linspace(1, 1, n_sampl)
    vals[:, 1] = np.linspace(1, 0, n_sampl)
    vals[:, 2] = np.linspace(0, 0, n_sampl)

    return ListedColormap(vals)


def recursive(filename, mtrcs, req_metrics,
              show=False, title=None, job_id=None, step_id=None):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `mtrcs` supported by ear_analytics.
    """
    def preprocess_df(data_f):
        """
        Pre-process DataFrame `data_f` to get workable data.
        """
        return (data_f
                .assign(
                  GPOWER=lambda x: round(data_f['GPOWER'] * 10**-6, 4),
                  avg_cpu_freq=lambda x:
                  round(data_f['AVG.CPUFREQ'] * 10**-6, 4),
                  avg_imc_freq=lambda x:
                  round(data_f['AVG.IMCFREQ'] * 10**-6, 4),
                  ENERGY_TAG=lambda x:
                  data_f['TIME'] * data_f['DC-NODE-POWER'],
                )
                .drop(['DEF.FREQ', 'AVG.CPUFREQ', 'AVG.IMCFREQ'], axis=1)
                )

    data_f = filter_by_job_step_app(read_data(filename), job_id=job_id, step_id=step_id)

    # Prepare x-axe range for iterations captured
    x_sampl = np.linspace(min(data_f.index.values),
                          max(data_f.index.values), dtype=int)
    extent = [x_sampl[0]-(x_sampl[1]-x_sampl[0])//2,
              x_sampl[-1]+(x_sampl[1]-x_sampl[0])//2, 0, 1]

    # Compute the heatmap graph for each metric specified by the input

    for metric in req_metrics:
        metric_name = mtrcs.get_metric(metric).name

        # m_data = group_by_node[metric_name].interpolate(method='bfill',
        # limit_area='inside')
        m_data = data_f[metric_name]  # .interpolate(method='bfill',
        # limit_area='inside')

        for key in x_sampl:
            if key not in m_data.index:
                m_data = m_data.append(pd.Series(name=key, dtype=object),
                                       verify_integrity=True)
                m_data.sort_index(inplace=True)

        m_data.interpolate(method='bfill', limit_area='inside')
        for idx in m_data.index.values:
            if idx not in x_sampl:
                m_data.drop(idx, inplace=True)

        m_data_array = m_data.values.transpose()  # \
        # .interpolate(method='bfill', limit_area='inside')\

        # Create the resulting figure for current metric
        fig = plt.figure(figsize=[17.2, 9.6])

        tit = metric_name
        if title is not None:
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
        if show:
            plt.show()
            plt.pause(0.001)
        else:
            name = f'recursion_{title}_{metric_name}.jpg'
            plt.savefig(fname=name, bbox_inches='tight')


def recursive_parser_action_closure(metrics):
    """
    Closure function used to return the action
    function when `recursive` sub-command is called.
    """

    def rec_parser_action(args):
        """ Action for `recursive` subcommand """
        recursive(args.input_file, metrics, args.metrics,
                  args.show, args.title)

    def print_in_build_proces(args):
        print("This functionality is still under development.")

    return print_in_build_proces


def res_parser_action(args):
    """ Action for `resume` subcommand """
    print(args)
    resume(args.input_file, args.base_freq, args.app_name,
           args.jobid, args.show, args.output, args.title)


def build_parser(metrics):
    """
    Given the used `metrics`,
    returns a parser to read and check command line arguments.
    """
    parser = argparse.ArgumentParser(prog='ear_analytics',
                                     description='High level support for read '
                                     'and visualize information files given by'
                                     ' EARL.')
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
    parser.add_argument('-j', '--jobid', type=int,
                            help='Sets the JOB ID you are working'
                            ' with.')

    subparsers = parser.add_subparsers(help='The two functionalities currently'
                                       ' supported by this program.',
                                       description='Type `ear_analytics '
                                       '<dummy_filename> {recursive,resume} -h'
                                       '` to get more info of each subcommand')

    # create the parser for the `recursive` command
    parser_rec = subparsers.add_parser('recursive',
                                       help='Generate a heatmap graph showing'
                                       ' the behaviour of some metrics monitor'
                                       'ed with EARL. The file must be outpute'
                                       'd by `eacct -r` command.')
    parser_rec.add_argument('-s', '--stepid', type=int,
                            help='Sets the STEP ID of the job you are working'
                            ' with.')
    parser_rec.add_argument('-m', '--metrics', nargs='+',
                            choices=list(metrics.metrics.keys()),
                            required=True, help='Specify which metrics you wan'
                            't to visualize.')
    parser_rec.set_defaults(func=recursive_parser_action_closure(metrics))

    # create the parser for the `resume` command
    parser_res = subparsers.add_parser('resume',
                                       help='Generate a resume about Energy'
                                       ' and Power save, and Time penalty of'
                                       ' an application monitored with EARL.')
    parser_res.add_argument('base_freq', help='Specify which'
                            ' frequency is used as base '
                            'reference for computing and showing'
                            ' savings and penalties in the figure.',
                            type=float)
    parser_res.add_argument('--app_name', help='Set the application name to'
                            ' get resume info.')
    parser_res.set_defaults(func=res_parser_action)

    return parser


def read_ini(filename):
    """
    Load the configuration file `filename`
    """

    def parse_float_tuple(in_str):
        """
        Given a tuple of two ints in str type, returns a tuple of two ints.
        """
        return tuple(float(k.strip()) for k in in_str[1:-1].split(','))

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


def main():
    """ Entry method. """

    # Read configuration file and init `metrics` data structure
    metrics = init_metrics(read_ini('config.ini'))
    # print(f'METRICS CONFIG\n{metrics.__str__()}')

    # create the top-level parser
    parser = build_parser(metrics)

    args = parser.parse_args()

    """
    if args.log:
        log_lvl = getattr(logging, args.log.upper(), None)
        if not isinstance(log_lvl, int):
            logging.warning(f'Invalid log level: {args.log}'
                            '\nSetting log level to default...')
            log_lvl = getattr(logging, 'INFO', None)
    else:
        log_lvl = getattr(logging, 'INFO', None)
    """

    args.func(args)


if __name__ == '__main__':
    main()
