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
                avg_gpu_memfreq=lambda x: x.filter(regex=r'GMEMFREQ\d')
                .mean(axis=1),
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
                  .pivot_table(values=metric_filter,
                               index='TIMESTAMP', columns='NODENAME')
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
        gpu_metric_regex_str = r'(GFREQ|GUTIL|GPOWER|GMEMFREQ|GMEMUTIL)(\d)'
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
                if title:  # We preserve the title got by the user
                    tit = f'{title}: {metric_name}'
                else:  # The default title: %metric-%job_id-%step_id
                    if job_id:
                        tit = '-'.join([tit, str(job_id)])
                        if step_id is not None:
                            tit = '-'.join([tit, str(step_id)])
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
            name = f'runtime_{metric_name}'
            if job_id:
                name = '-'.join([name, str(job_id)])
                if step_id is not None:
                    name = '-'.join([name, str(step_id)])

            if output:
                if os.path.isdir(output):

                    name = os.path.join(output, name)
                else:
                    name = output

            print(f'storing figure at {name}.png')

            plt.savefig(fname=name, bbox_inches='tight', transparent=True)


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
              TIME=lambda df: df.TIME * 1000000,
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
              {'TIME': np.int64,
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
        df.loc[(df['JOBID'] == app_job) &
               (df['STEPID'] == app_step), 'app_id'] = np.int64(appl_idx + 1)

        for node_idx, node_name in enumerate(appl_nodes):
            df.loc[(df['JOBID'] == app_job) &
                   (df['STEPID'] == app_step) &
                   (df['NODENAME'] == node_name), 'task_id'] \
                           = np.int64(node_idx + 1)

            task_lvl_names = '\n'.join([task_lvl_names,
                                       f'({np.int64(node_idx + 1)})'
                                        f' {node_name}'])

            for gpu_idx in range(n_gpus):
                thread_lvl_names += f'({node_name}) GPU{gpu_idx}\n'

        # APPL level names
        appl_lvl_names = '\n'.join([appl_lvl_names,
                                   f'({np.int64(appl_idx + 1)}) ' +
                                   df_app['app_name'].unique()[0]])

    names_conf_str = '\n'.join([appl_lvl_names, task_lvl_names])

    if n_threads != 0:
        # Some application has GPUs, so we can configure the THREAD level
        thread_lvl_names = '\n'.join(['LEVEL THREAD SIZE'
                                      f' {n_threads * n_nodes}',
                                      thread_lvl_names])

        names_conf_str = '\n'.join([names_conf_str, thread_lvl_names])

    if not output_fn:
        output_fn = loop_data_fn.split('.')[0]

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

    cols_regex = re.compile(r'((GPOWER|GFREQ|GMEMFREQ|GUTIL|GMEMUTIL)(\d))'
                            r'|JOBID|STEPID|NODENAME|FIRST_EVENT|LEVEL|SIZE'
                            r'|TIMESTAMP|start_time|end_time|time|task_id'
                            r'|app_id|app_name')
    metrics = df.drop(columns=df.filter(regex=cols_regex).columns).columns

    event_typ_lst = []
    for metric in metrics:
        event_typ_lst.append(f'EVENT_TYPE\n0\t'
                     f'{trace_sorted_df.columns.get_loc(metric)}\t'
                     f'{metric}\n')

    event_typ_lst.append('\n')

    event_typ_str = '\n'.join(event_typ_lst)

    paraver_conf_file_str = '\n'.join([def_options_str, event_typ_str])

    # print(f'Paraver configuration file:\n{paraver_conf_file_str}')

    with open('.'.join([output_fn, 'pcf']), 'w') as pcf_file:
        pcf_file.write(paraver_conf_file_str)


def eacct(result_format, jobid, stepid):
    # A temporary folder to store the generated csv file
    csv_file = '.'.join(['_'.join(['tmp', str(jobid), str(stepid)]), 'csv'])

    if result_format == "runtime":
        os.system(f"eacct -j {jobid}.{stepid} -r -c {csv_file}")
    elif result_format == "ear2prv":
        os.system(f"eacct -j {jobid}.{stepid} -r -o -c {csv_file}")
    else:
        print("Unrecognized format: Please contact with support@eas4dc.com")

    return csv_file


def parser_action_closure(conf_metrics):

    def parser_action(args):

        csv_generated = False

        if args.input_file is None:
            # Action performing eacct command and storing csv files
            input_file = eacct(args.format, args.jobid, args.stepid)
            args.input_file = input_file
            csv_generated = True

        if args.format == "runtime":
            runtime(args.input_file, conf_metrics, args.metrics,
                    args.relative_range, args.save, args.title, args.jobid,
                    args.stepid, args.output, args.horizontal_legend)

        if args.format == "ear2prv":
            head_path, tail_path = os.path.split(args.input_file)
            out_jobs_path = os.path.join(head_path,
                                         '.'.join(['out_jobs', tail_path]))
            ear2prv(out_jobs_path, args.input_file, job_id=args.jobid,
                    step_id=args.stepid, output_fn=args.output)

        if csv_generated and not args.keep_csv:
            os.system(f'rm {input_file}')
            if args.format == 'ear2prv':
                os.system(f'rm {out_jobs_path}')

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
                    choices=['runtime', 'ear2prv'],
                    help='Build results according to chosen format: '
                    'runtime (static images) or ear2prv (using paraver '
                    'tool).')

    parser.add_argument('--input_file', help='Specifies the input file(s) '
                                         'name(s) to read data from.')

    parser.add_argument('-j', '--jobid', type=int, required=True,
                        help='Filter the data by the Job ID.')
    parser.add_argument('-s', '--stepid', type=int, required=True,
                        help='Filter the data by the Step ID.')

    # ONLY for runtime format
    runtime_group_args = parser.add_argument_group('`runtime` format options')

    group = runtime_group_args.add_mutually_exclusive_group()
    group.add_argument('--save', action='store_true',
                       help='Activate the flag to store resulting figures.')
    group.add_argument('--show', action='store_true',
                       help='Show the resulting figure (default).')

    runtime_group_args.add_argument('-t', '--title',
                                help='Set the resulting figure title '
                                '(Only valid with runtime format).')

    runtime_group_args.add_argument('-r', '--relative_range',
                                    action='store_true',
                                    help='Use the relative range of a metric '
                                    'over the trace data to build the '
                                    'gradient.')

    runtime_group_args.add_argument('-l', '--horizontal_legend',
                                    action='store_true',
                                    help='Display the legend horizontally. '
                                    'This option is useful when your trace has'
                                    ' a low number of nodes.')

    runtime_group_args.add_argument('-m', '--metrics', nargs='+',
                                    choices=list(conf_metrics.metrics.keys()),
                                    help='Space separated list of case sensitive'
                                    ' metrics names to visualize. Allowed values are '
                                    f'{", ".join(conf_metrics.metrics.keys())}',
                                    metavar='metric')

    parser.add_argument('-o', '--output',
                        help='Sets the output name. You can just set a path or'
                             ' a filename. For `runtime` format option, this '
                             'argument is only valid if `--save` flag is'
                             ' given.')

    parser.add_argument('-k', '--keep_csv', action='store_true',
                        help="Don't remove temorary csv files.")

    parser.set_defaults(func=parser_action_closure(conf_metrics))

    return parser


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
