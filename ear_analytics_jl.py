""" High level support for read and visualize
    information given by EARL. """

import argparse
import os
import time
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import colorcet as cc

from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from common.io_api import read_data, read_ini
from common.metrics import init_metrics
from common.utils import filter_df


def resume(filename, base_freq, app_id=None, job_id=None,
           save=False, output=None, title=None):
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
    data_f = (read_data(filename)
              .pipe(filter_df, JOBID=job_id, APP_ID=app_id)  # Filter rows
              .pipe(preprocess_df)
              )
    # data_f = preprocess_df(filter_by_job_step_app(read_data(filename),
    #                        job_id=job_id, app_id=app_id))

    # Compute per step energy consumed
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
    # freqs = dropped[['avg_cpu_freq', 'avg_imc_freq']]

    # Prepare and create the plot
    tit = 'resume'
    if title:
        tit = title
    elif app_id:
        tit = app_id + f' vs. {base_freq} GHz'

    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')

    axes = results.plot(kind='bar', figsize=(8, 6),
                        rot=0, legend=False)  # , fontsize=20)
    axes.set_xlabel('POLICY, def. Freq (GHz)')  # , fontsize=20)
    axes.set_title(tit, loc='center', wrap=True, pad=10.0, weight='bold')
    plt.tight_layout()

    # plt.gcf().suptitle(tit)  # , fontsize='22', weight='bold')

    # ax2 = axes.twinx()
    # freqs.plot(ax=ax2,  ylim=(0, 3.5), color=['cyan', 'purple'],
    #            linestyle='-.', legend=False, fontsize=20)
    # ax2.set_ylabel(ylabel='avg. Freq (GHz)', labelpad=20.0, fontsize=20)

    # create the legend
    handles_1, labels_1 = axes.get_legend_handles_labels()
    # handles_2, labels_2 = ax2.get_legend_handles_labels()

    # axes.legend(handles_1 + handles_2,
    # labels_1 + labels_2, loc=0, fontsize=15)
    axes.legend(handles_1, labels_1, loc=0)  # , fontsize=15)

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
                  ha='center', va='bottom')  # , fontsize=12)
    if not save:
        plt.show()
    else:
        name = 'resume.jpg'
        if output is not None:
            name = output
        plt.savefig(fname=name, bbox_inches='tight')


def runtime(filename, mtrcs, req_metrics, rel_range=False, save=False,
            title=None, job_id=None, step_id=None, output=None,
            horizontal_legend=False):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `mtrcs` supported by ear_analytics.
    """

    def join_metric_node(df):
        "Given a DataFrame df, returns it flattening it's columns MultiIndex."
        df.columns = df.columns.to_flat_index()
        return df

    df = (read_data(filename)
          .pipe(filter_df, JOBID=job_id, STEPID=step_id, JID=job_id)
          # .groupby(['NODENAME', 'TIMESTAMP'])
          # .agg(lambda x: x).unstack(level=0)
          .assign(
                avg_gpu_pwr=lambda x: x.filter(regex=r'GPOWER\d').mean(axis=1),
                tot_gpu_pwr=lambda x: x.filter(regex=r'GPOWER\d').sum(axis=1),
                avg_gpu_freq=lambda x: x.filter(regex=r'GFREQ\d').mean(axis=1),
                tot_gpu_freq=lambda x: x.filter(regex=r'GFREQ\d').sum(axis=1),
                avg_gpu_memfreq=lambda x: x.filter(regex=r'GMEMFREQ\d')
                .mean(axis=1),
                tot_gpu_memfreq=lambda x: x.filter(regex=r'GMEMFREQ\d')
                .sum(axis=1),
                avg_gpu_util=lambda x: x.filter(regex=r'GUTIL\d').mean(axis=1),
                tot_gpu_util=lambda x: x.filter(regex=r'GUTIL\d').sum(axis=1),
                avg_gpu_memutil=lambda x: x.filter(regex=r'GMEMUTIL\d')
                .mean(axis=1),
                tot_gpu_memutil=lambda x: x.filter(regex=r'GMEMUTIL\d')
                .sum(axis=1),
              )
          )

    # Prepare x-axe range for iterations captured

    for metric in req_metrics:
        metric_name = mtrcs.get_metric(metric).name

        metric_filter = df.filter(regex=metric_name).columns

        m_data = (df
                  .pivot_table(values=metric_filter, index='TIMESTAMP', columns='NODENAME')
                  .bfill()
                  .pipe(join_metric_node)
                  )
        m_data.index = pd.to_datetime(m_data.index, unit='s')

        new_idx = pd.date_range(start=m_data.index[0], end=m_data.index[-1],
                                freq='10S').union(m_data.index)

        m_data = m_data.reindex(new_idx).bfill()

        m_data_array = m_data.values.transpose()

        x_lim = mdates.date2num([m_data.index.min(), m_data.index.max()])

        # Create the resulting figure for current metric
        fig = plt.figure(figsize=[19.2, 0.5 * len(m_data.columns)])

        # We use a grid layout to easily insert the gradient legend
        if not horizontal_legend:
            grid_sp = GridSpec(nrows=len(m_data_array), ncols=2,
                               width_ratios=(9.5, 0.5), hspace=0, wspace=0.04)
        else:
            grid_sp = GridSpec(nrows=len(m_data_array) + 1, ncols=1, hspace=1)

            gs1 = GridSpecFromSubplotSpec(len(m_data_array), 1,
                                          subplot_spec=grid_sp[0:-1])
            gs2 = GridSpecFromSubplotSpec(1, 1, subplot_spec=grid_sp[-1])

        norm = mtrcs.get_metric(metric).norm_func()  # Absolute range
        if rel_range:  # Relative range
            norm = Normalize(vmin=np.nanmin(m_data_array),
                             vmax=np.nanmax(m_data_array), clip=True)
            # print(f'Using a gradient range of ({np.nanmin(m_data_array)}, '
            #       f'{np.nanmax(m_data_array)}) for {metric_name}')

        # Check if the requested metric is per GPU
        gpu_metric_regex_str = r'(GFREQ|GUTIL|GFREQ|GMEMFREQ|GMEMUTIL)(\d)'
        gpu_metric_regex = re.compile(gpu_metric_regex_str)

        for i, _ in enumerate(m_data_array):

            gpu_metric_match = gpu_metric_regex.search(m_data.columns[i][0])

            if gpu_metric_match:
                ylabel_text = (f'GPU{gpu_metric_match.group(2)}'
                               f' @ {m_data.columns[i][1]}')
            else:
                ylabel_text = m_data.columns[i][1]

            if not horizontal_legend:
                axes = fig.add_subplot(grid_sp[i, 0], ylabel=ylabel_text)
            else:
                axes = fig.add_subplot(gs1[i], ylabel=ylabel_text)

            axes.set_yticks([])
            axes.set_ylabel(ylabel_text, rotation=0,
                            weight='bold', labelpad=len(ylabel_text) * 4)

            data = np.array(m_data_array[i], ndmin=2)

            axes.imshow(data, cmap=ListedColormap(list(reversed(cc.bgy))),
                        norm=norm, aspect='auto',
                        extent=[x_lim[0], x_lim[1], 0, 1])
            axes.set_xlim(x_lim[0], x_lim[1])

            date_format = mdates.DateFormatter('%x: %H:%M:%S')
            axes.xaxis.set_major_formatter(date_format)

            if i == 0:
                tit = metric_name
                if title is not None:
                    tit = f'{title}: {metric_name}'
                axes.set_title(tit, weight='bold')

            if i < len(m_data_array) - 1:
                axes.set_xticklabels([])

        if horizontal_legend:
            plt.subplots_adjust(hspace=0.0)
            col_bar_ax = fig.add_subplot(gs2[0, 0])
            fig.colorbar(cm.ScalarMappable(
                cmap=ListedColormap(list(reversed(cc.bgy))), norm=norm),
                cax=col_bar_ax, orientation="horizontal")
        else:
            col_bar_ax = fig.add_subplot(grid_sp[:, 1])
            fig.colorbar(cm.ScalarMappable(
                cmap=ListedColormap(list(reversed(cc.bgy))), norm=norm),
                cax=col_bar_ax)

        if not save:
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


def ear2prv(job_data_fn, loop_data_fn, job_id=None,
            step_id=None, output_fn=None):

    # Read the data

    df_job = (read_data(job_data_fn)
              .pipe(filter_df, id=job_id, step_id=step_id)
              .pipe(lambda df: df[['id', 'step_id', 'app_id',
                                   'start_time', 'end_time']])
              .pipe(lambda df: df.rename(columns={"id": "JOBID",
                                                  "step_id": "STEPID",
                                                  'app_id': 'app_name'}))
              )
    # print('Job data:\n', df_job)

    df = (read_data(loop_data_fn)
          .pipe(filter_df, JOBID=job_id, STEPID=step_id)
          .merge(df_job)
          .assign(
              CPI=lambda df: df.CPI * 1000000,  # Paraver needs values > 0
              ITER_TIME_SEC=lambda df: df.ITER_TIME_SEC * 1000000,
              IO_MBS=lambda df: df.IO_MBS * 1000000,
              time=lambda df: (df.TIMESTAMP - df.start_time) * 1000000  # (us)
              )
          .join(pd.Series(dtype=np.int64, name='task_id'))
          .join(pd.Series(dtype=np.int64, name='app_id'))
          .join(pd.Series(dtype=np.int64, name='gpu_power'))
          .join(pd.Series(dtype=np.int64, name='gpu_freq'))
          .join(pd.Series(dtype=np.int64, name='gpu_mem_freq'))
          .join(pd.Series(dtype=np.int64, name='gpu_util'))
          .join(pd.Series(dtype=np.int64, name='gpu_mem_util'))
          .astype(
              {'ITER_TIME_SEC': np.int64,
               'CPI': np.int64,
               'TPI': np.int64,
               'MEM_GBS': np.int64,
               'IO_MBS': np.int64,
               'PERC_MPI': np.int64,
               'DC_NODE_POWER_W': np.int64,
               'DRAM_POWER_W': np.int64,
               'PCK_POWER_W': np.int64,
               'GFLOPS': np.int64,
               })
          )
    # print('Working DataFrame:\n', df)

    # Generating the Application list and row file

    node_info = pd.unique(df.NODENAME)
    n_nodes = node_info.size

    # If only one step is assumed, the usage of numpy methods can be avoided.
    f_time = (np.max(df_job.end_time) - np.min(df_job.start_time)) * 1000000
    # print(f'Your trace files have {n_nodes} node(s) and they have a duration'
    #       f' time of {f_time} seconds.')

    """
    print('Nodes are:')
    for i, n in enumerate(node_info):
        print(f'{i + 1}) {n}')
    """

    appl_info = df.groupby(['JOBID', 'STEPID']).indices
    n_appl = len(appl_info)

    appl_list_str = ''  # The resulting Application List

    # The task level names section (.row file) can be created here
    task_lvl_names = f'LEVEL TASK SIZE {n_nodes * n_appl}'

    # The application level names section (.row) can be created here
    appl_lvl_names = f'LEVEL APPL SIZE {n_appl}'

    # Thread level, used to tag GPUs
    thread_lvl_names = ''
    n_threads = 0

    for appl_idx, (app_job, app_step) in enumerate(appl_info):
        df_app = df[(df['JOBID'] == app_job) & (df['STEPID'] == app_step)]

        appl_nodes = df['NODENAME'].unique()
        n_tasks = appl_nodes.size

        gpu_info = df_app.filter(regex=r'GPOWER').columns
        n_gpus = max(gpu_info.size, 1)

        # We accumulate the number of GPUs (threads)
        # associated to this node (task)
        n_threads += gpu_info.size

        # print(f'{appl_idx + 1}) {app_job}-{app_step}: {n_tasks} '
        #       f'task(s), nodes {appl_nodes}, {n_gpus} GPUs')

        appl_list = [f'{n_gpus}:{node_idx + 1}'
                     for node_idx, _ in enumerate(appl_nodes)]
        appl_list_str = ''.join([appl_list_str,
                                 f'{n_tasks}({",".join(appl_list)})'])

        # print(f'Application {appl_idx + 1} list: {appl_list_str}')

        # TASK level names
        df.loc[(df['JOBID'] == app_job)
               & (df['STEPID'] == app_step), 'app_id'] = np.int64(appl_idx + 1)

        for node_idx, node_name in enumerate(appl_nodes):
            df.loc[(df['JOBID'] == app_job)
                   & (df['STEPID'] == app_step)
                   & (df['NODENAME'] == node_name), 'task_id'] \
                           = np.int64(node_idx + 1)

            task_lvl_names = '\n'.join([task_lvl_names,
                                       f'({np.int64(node_idx + 1)})'
                                        f' {node_name}'])

            for gpu_idx in range(n_gpus):
                thread_lvl_names += f'({node_name}) GPU{gpu_idx}\n'

        # APPL level names
        appl_lvl_names = '\n'.join([appl_lvl_names,
                                   f'({np.int64(appl_idx + 1)}) '
                                    + df_app['app_name'].unique()[0]])

    names_conf_str = '\n'.join([appl_lvl_names, task_lvl_names])

    if n_threads != 0:
        # Some application has GPUs, so we can configure the THREAD level
        thread_lvl_names = '\n'.join(['LEVEL THREAD SIZE'
                                      f' {n_threads * n_nodes}',
                                      thread_lvl_names])

        names_conf_str = '\n'.join([names_conf_str, thread_lvl_names])

    with open('.'.join([output_fn, 'row']), 'w') as row_file:
        # print(f'Row file:\n{names_conf_str}')
        row_file.write(names_conf_str)

    # Generating the header
    date_time = time.strftime('%d/%m/%y at %H:%M',
                              time.localtime(np.min(df_job.start_time)))
    file_trace_hdr = (f'#Paraver ({date_time}):{f_time}'
                      f':0:{n_appl}:{appl_list_str}')
    # print(f'Paraver trace header: {file_trace_hdr}')

    # Trace file

    metrics = (df.drop(columns=['JOBID', 'STEPID', 'NODENAME', 'FIRST_EVENT',
                                'LEVEL', 'SIZE', 'TIMESTAMP', 'start_time',
                                'end_time', 'time', 'task_id', 'app_id',
                                'app_name', 'gpu_power', 'gpu_freq',
                                'gpu_mem_freq', 'gpu_util', 'gpu_mem_util'])
                 .columns)

    # We first sort data by timestamp in ascending
    # order as specified by Paraver trace format.
    trace_sorted_df = df.sort_values('time')

    records = trace_sorted_df.to_records(index=False)
    columns = trace_sorted_df.columns

    app_id_idx = columns.get_loc('app_id')
    task_id_idx = columns.get_loc('task_id')
    timestamp_idx = columns.get_loc('time')

    gpu_field_regex = re.compile(r'(GPOWER|GFREQ|GMEMFREQ|GUTIL|GMEMUTIL)(\d)')
    gpu_field_map = {'GPOWER': 'gpu_power', 'GFREQ': 'gpu_freq',
                     'GMEMFREQ': 'gpu_mem_freq', 'GUTIL': 'gpu_util',
                     'GMEMUTIL': 'gpu_mem_util',
                     }

    body_list = []
    for row in records:
        for metric in metrics:
            # Get the column index of the metric
            metric_idx = columns.get_loc(metric)
            event_val = np.int64(row[metric_idx])

            # The default thread index. For non-GPU fields it won't be used.
            thread_idx = 1

            # Check if the metric is related with the GPU
            gpu_field = gpu_field_regex.search(metric)
            if gpu_field:
                # Update the thread id based on the GPU number
                thread_idx = int(gpu_field.group(2)) + 1

                metric_idx = columns.get_loc(gpu_field_map[gpu_field.group(1)])

            body_list.append(f'2:0:{"{:0.0f}".format(row[app_id_idx])}'
                             f':{"{:0.0f}".format(row[task_id_idx])}'
                             f':{thread_idx}:{row[timestamp_idx]}'
                             f':{metric_idx}:{event_val}')

    file_trace_body = '\n'.join(body_list)

    with open('.'.join([output_fn, 'prv']), 'w') as prv_file:
        prv_file.write('\n'.join([file_trace_hdr, file_trace_body]))

    # Paraver Configuration file
    def_options_str = 'DEFAULT_OPTIONS\n\nLEVEL\tTASK\nUNITS\tSEC\n'

    cols_regex = re.compile(r'((GPOWER|GFREQ|GMEMFREQ|GUTIL|GMEMUTIL)(\d))|JOBID|STEPID|NODENAME|FIRST_EVENT|LEVEL|SIZE|TIMESTAMP|start_time|end_time|time|task_id|app_id|app_name')
    metrics = df.drop(columns=df.filter(regex=cols_regex).columns).columns

    event_typ_lst = []
    for metric in metrics:
        event_typ_lst.append(f'EVENT_TYPE\n0\t{trace_sorted_df.columns.get_loc(metric)}\t{metric}\n')

    event_typ_lst.append('\n')

    event_typ_str = '\n'.join(event_typ_lst)

    paraver_conf_file_str = '\n'.join([def_options_str, event_typ_str])

    # print(f'Paraver configuration file:\n{paraver_conf_file_str}')

    with open('.'.join([output_fn, 'pcf']), 'w') as pcf_file:
        pcf_file.write(paraver_conf_file_str)


def runtime_parser_action_closure(metrics):
    """
    Closure function used to return the action
    function when `recursive` sub-command is called.
    """

    def run_parser_action(args):
        """ Action for `recursive` subcommand """
        runtime(args.input_file, metrics, args.metrics, args.relative_range,
                args.save, args.title, args.jobid, args.stepid, args.output,
                args.horizontal_legend)

    return run_parser_action


def res_parser_action(args):
    """ Action for `resume` subcommand """
    resume(args.input_file, args.base_freq, args.app_name,
           args.jobid, args.save, args.output, args.title)


def prv_parser_action(args):
    """ Action for `ear2prv` subcommand """
    head_path, tail_path = os.path.split(args.input_file)
    # print(head_path, tail_path)
    out_jobs_path = os.path.join(head_path, '.'.join(['out_jobs', tail_path]))
    ear2prv(out_jobs_path, args.input_file, job_id=args.jobid,
            step_id=args.step_id, output_fn=args.output)


def parser_action_closure(conf_metrics):

    def parser_action(args):
        print("in parser_action:\n",args)

        csv_generated = False

        if args.input_file == None:
            # Action performing eacct command and storing csv files
            input_file = eacct(args.format, args.jobid, args.stepid)
            args.input_file = input_file
            csv_generated = True

        if args.format == "runtime" :
            print("args input_file = ", args.input_file)
            runtime(args.input_file, conf_metrics, args.metrics, args.relative_range,
            args.save, args.title, args.jobid, args.stepid, args.output,
            args.horizontal_legend)

        if args.format == "ear2prv" :
            head_path, tail_path = os.path.split(args.input_file)
            out_jobs_path = os.path.join(head_path, '.'.join(['out_jobs', tail_path]))
            ear2prv(out_jobs_path, args.input_file, job_id=args.jobid, step_id=args.stepid, output_fn=args.output)

        if csv_generated and not args.keep_csv:
            os.system("rm " + input_file)

    return parser_action


def build_parser(conf_metrics):
    """
    Given the used `conf_metrics`, returns a parser to
    read and check command line arguments.
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
    parser.add_argument('--version', action='version', version='%(prog)s 4.0')

    parser.add_argument('--format', required=True,
            help='Build results according to chosen format: runtime or ear2prv '
                 '(using paraver tool).')

    parser.add_argument('--input_file', help='Specifies the input file(s) name(s'
                        ') to read data from.')

    parser.add_argument('-j', '--jobid', type=int, required=True,
                        help='Filter the data by the Job ID.')
    parser.add_argument('--stepid', type=int, required=True,
                        help='Filter the data by the Step ID.')

    parser.add_argument('-m', '--metrics', nargs='+',
                            choices=list(conf_metrics.metrics.keys()),
                            help='Space separated list of case sensitive'
                            ' metrics names to visualize. Allowed values are '
                            + ', '.join(conf_metrics.metrics.keys()),
                            metavar='metric')

    # ONLY for runtime format
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save', action='store_true',
                       help='Activate the flag to store resulting figures.')
    group.add_argument('--show', action='store_true',
                       help='Show the resulting figure (default).')
    parser.add_argument('-t', '--title',
                        help='Set the resulting figure title '
                             '(Only valid with runtime format).')
    parser.add_argument('-o', '--output',
                        help='Sets the output image name.'
                             'Only valid if `--save` flag is set '
                            ' and with runtime format')
    parser.add_argument('-r', '--relative_range', action='store_true',
                         help='Use the relative range of a metric over the '
                         'trace data to build the gradient.')
    parser.add_argument('-l', '--horizontal_legend', action='store_true',
                            help='Display the legend horizontally. This option'
                            ' is useful when your trace has a low number of'
                            ' nodes.')

    parser.add_argument('--keep_csv', action='store_true',
                            help='remove temorary csv file')

    parser.set_defaults(func=parser_action_closure(conf_metrics))

    return parser


def eacct(result_format, jobid, stepid):
    # A temporary folder to store the generated csv file
    print("in aacct")
    csv_file = "tmp_"+str(jobid)+"."+str(stepid)+".csv"

    if result_format == "runtime":
        os.system("eacct -j " + str(jobid) + "." + str(stepid) + " -r -c " + csv_file)
    elif result_format == "ear2prv":
        os.system("eacct -j " + str(jobid) + "." + str(stepid) + " -r -o -c " + csv_file)
    else:
        print("Unrecognized format: please choose between 'runtime' and 'ear2prv'.")

    return csv_file

def main():
    """ Entry method. """

    # Read configuration file and init `metrics` data structure
    conf_metrics = init_metrics(read_ini('config.ini'))

    # create the top-level parser
    parser = build_parser(conf_metrics)

    args = parser.parse_args()

    # condition if input file not given
    args.func(args)


if __name__ == '__main__':
    main()
