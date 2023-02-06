""" High level support for read and visualize
    information given by EARL. """

import argparse
import os
import sys
import subprocess
import time
import re
import json

import pandas as pd
import numpy as np
import colorcet as cc
import proplot as pplt

from heapq import merge

from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize

from common.io_api import read_data
from common.metrics import read_metrics_configuration, metric_regex
from common.utils import filter_df, join_metric_node

from common.phases import (read_phases_configuration,
                           df_phases_phase_time_ratio,
                           df_phases_to_tex_tabular)

from common.job_summary import (job_cpu_summary_df, job_summary_to_tex_tabular,
                                job_gpu_summary)
from common.ear_data import df_has_gpu_data, df_get_valid_gpu_data


def job_summary(df_long, df_loops, df_phases, metrics_conf, phases_conf):
    """
    Generate a job summary.
    """
    job_id = df_long['JOBID'].unique()
    if job_id.size != 1:
        print('ERROR: Only one job is supported. Jobs detected: {job_id}.')
        return
    else:
        job_id = str(job_id[0])

    try:
        print(f'Creating directory {job_id}')
        os.mkdir(job_id)
    except FileExistsError:
        print(f'Directory {job_id} already exists.')
        return

    tables_dir = os.path.join(job_id, 'tables')

    print(f'Creating {tables_dir} directory')
    os.mkdir(tables_dir)

    # Job summary
    (job_cpu_summary_df(df_long, metrics_conf)
     .pipe(job_summary_to_tex_tabular,
           os.path.join(tables_dir, 'job_summary')))

    # Job summary (GPU part)
    (job_gpu_summary(df_long)
     .pipe(job_summary_to_tex_tabular, os.path.join(tables_dir,
                                                    'job_gpu_summary')))

    timelines_dir = os.path.join(job_id, 'timelines')
    os.mkdir(timelines_dir)

    # Aggregated power
    agg_metric_timeline(df_loops, metric_regex('dc_power', metrics_conf),
                        os.path.join(timelines_dir, 'agg_dcpower'),
                        fig_title='Aggregated DC Node Power (W)')

    # Aggregated Mem. bandwidth
    agg_metric_timeline(df_loops, metric_regex('gbs', metrics_conf),
                        os.path.join(timelines_dir, 'agg_gbs'),
                        fig_title='Aggregated memory bandwidth (GB/s)')

    # Aggregated GFlop/s
    agg_metric_timeline(df_loops, metric_regex('gflops', metrics_conf),
                        os.path.join(timelines_dir, 'agg_gflops'),
                        fig_title='Aggregated CPU GFlop/s')

    # Aggregated I/O
    agg_metric_timeline(df_loops, metric_regex('io_mbs', metrics_conf),
                        os.path.join(timelines_dir, 'agg_iombs'),
                        fig_title='Aggregated I/O throughput (MB/s)')

    # GPU timelines
    if df_has_gpu_data(df_loops):

        # Aggregated GPU power
        gpu_pwr_re = metric_regex('gpu_power', metrics_conf)

        df_agg_gpupwr = (df_loops
                         .assign(
                             tot_gpu_pwr=lambda x: (x.filter(regex=gpu_pwr_re)
                                                     .sum(axis=1)
                                                    )
                             )
                         )
        agg_metric_timeline(df_agg_gpupwr, 'tot_gpu_pwr',
                            os.path.join(timelines_dir, 'agg_gpupower'),
                            fig_title='Aggregated GPU Power (W)')

        # Per-node GPU util
        norm = Normalize(vmin=0, vmax=100, clip=True)
        metric_timeline(filter_invalid_gpu_series(df_loops),
                        metric_regex('gpu_util', metrics_conf),
                        os.path.join(timelines_dir, 'per-node_gpuutil'),
                        norm=norm, fig_title='GPU utilization (%)')

    # Per-node CPI
    metric_timeline(df_loops, metric_regex('cpi', metrics_conf),
                    os.path.join(timelines_dir, 'per-node_cpi'),
                    fig_title='Cycles per Instruction')

    # Per-node GBS
    metric_timeline(df_loops, metric_regex('gbs', metrics_conf),
                    os.path.join(timelines_dir, 'per-node_gbs'),
                    fig_title='Memory bandwidth (GB/s)')

    # Per-node GFlop/s
    metric_timeline(df_loops, metric_regex('gflops', metrics_conf),
                    os.path.join(timelines_dir, 'per-node_gflops'),
                    fig_title='CPU GFlop/s')

    # Per-node Avg. CPU freq.
    metric_timeline(df_loops, metric_regex('avg_cpufreq', metrics_conf),
                    os.path.join(timelines_dir, 'per-node_avgcpufreq'),
                    fig_title='Avg. CPU frequency (kHz)')

    # Phases summary
    (df_phases
     .pivot(index='node_id', columns='Event_type', values='Value')
     .pipe(df_phases_phase_time_ratio, phases_conf)
     .pipe(df_phases_to_tex_tabular, 'job_phases_summary'))


def df_gpu_node_metrics(df, conf_fn='config.json'):
    """
    Given a DataFrame `df` with EAR data and a configuration filename `conf_fn`
    Returns a copy of the DataFrame with new columns showing node-level GPU
    metrics.
    """
    metrics_conf = read_metrics_configuration(conf_fn)

    gpu_pwr_regex = metric_regex('gpu_power', metrics_conf)
    gpu_freq_regex = metric_regex('gpu_freq', metrics_conf)
    gpu_memfreq_regex = metric_regex('gpu_memfreq', metrics_conf)
    gpu_util_regex = metric_regex('gpu_util', metrics_conf)
    gpu_memutil_regex = metric_regex('gpu_memutil', metrics_conf)

    return (df
            .assign(
                tot_gpu_pwr=lambda x: (x.filter(regex=gpu_pwr_regex)
                                        .sum(axis=1)),  # Agg. GPU power

                avg_gpu_pwr=lambda x: (x.filter(regex=gpu_pwr_regex)
                                        .mean(axis=1)),  # Avg. GPU power

                avg_gpu_freq=lambda x: (x.filter(regex=gpu_freq_regex)
                                        .mean(axis=1)),  # Avg. GPU freq

                avg_gpu_memfreq=lambda x: (x.filter(regex=gpu_memfreq_regex)
                                           .mean(axis=1)),  # Avg. GPU mem freq

                avg_gpu_util=lambda x: (x.filter(regex=gpu_util_regex)
                                        .mean(axis=1)),  # Avg. % GPU util

                avg_gpu_memutil=lambda x: (x.filter(regex=gpu_memutil_regex)
                                           .mean(axis=1)),  # Avg %GPU mem util
                ))


def filter_invalid_gpu_series(df):
    """
    Given a DataFrame with EAR data, filters those GPU
    columns that not contain some of the job's GPUs used.

    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """
    gpu_metric_regex_str = (r'GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                            r'UTIL_PERC|MEM_UTIL_PERC)')

    return (df
            .drop(df  # Erase GPU columns
                  .filter(regex=gpu_metric_regex_str).columns, axis=1)
            .join(df  # Join with valid GPU columns
                  .pipe(df_get_valid_gpu_data),
                  validate='one_to_one'))  # Validate the join operation


def metric_timeseries_by_node(df, metric):
    """
    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """
    return (df
            .pivot_table(values=metric,
                         index='TIMESTAMP', columns='NODENAME')
            .bfill()
            .pipe(join_metric_node)
            )


def metric_agg_timeseries(df, metric):
    """
    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """
    return(df
           .pivot_table(values=metric,
                        index='TIMESTAMP', columns='NODENAME')
           .bfill()
           .ffill()
           .pipe(join_metric_node)
           .agg(np.sum, axis=1)
           )


def generate_metric_timeline_fig(df, metric, norm=None, fig_title='',
                                 vertical_legend=False, granularity='node'):
    """
    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """

    metric_filter = df.filter(regex=metric).columns

    if granularity != 'app':
        m_data = metric_timeseries_by_node(df, metric_filter)
    else:
        m_data = metric_agg_timeseries(df, metric_filter)

    m_data.index = pd.to_datetime(m_data.index, unit='s')

    new_idx = pd.date_range(start=m_data.index[0], end=m_data.index[-1],
                            freq='1S').union(m_data.index)

    m_data = m_data.reindex(new_idx).bfill()

    m_data_array = m_data.values.transpose()
    if granularity == 'app':
        m_data_array = m_data_array.reshape(1, m_data_array.shape[0])

    # Compute time deltas to be showed to
    # the end user instead of an absolute timestamp.
    time_deltas = [i - m_data.index[0] for i in m_data.index]

    # Create the resulting figure for current metric

    fig = pplt.figure(sharey=False, refaspect=20, suptitle=fig_title)

    if vertical_legend:
        grid_sp = pplt.GridSpec(nrows=len(m_data_array), ncols=2,
                                width_ratios=(0.95, 0.05), hspace=0)
    else:
        def metric_row(i):
            """
            returns whether row i corresponds to a metric timeline.
            """
            return i < len(m_data_array)

        height_ratios = [0.8 / len(m_data_array)
                         if metric_row(i) else 0.2
                         for i in range(len(m_data_array) + 1)]

        hspaces = [0 if metric_row(i + 1) else None
                   for i in range(len(m_data_array))]

        grid_sp = pplt.GridSpec(nrows=len(m_data_array) + 1, ncols=1,
                                hratios=height_ratios,
                                hspace=hspaces)

    # Normalize values

    if norm is None:  # Relative range
        norm = Normalize(vmin=np.nanmin(m_data_array),
                         vmax=np.nanmax(m_data_array), clip=True)

    gpu_metric_regex_str = (r'GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                            r'UTIL_PERC|MEM_UTIL_PERC)')
    gpu_metric_regex = re.compile(gpu_metric_regex_str)

    for i, _ in enumerate(m_data_array):
        if granularity != 'app':
            gpu_metric_match = gpu_metric_regex.search(m_data.columns[i][0])

            if gpu_metric_match:
                ylabel_text = (f'GPU{gpu_metric_match.group(1)}'
                               f' @ {m_data.columns[i][1]}')
            else:
                ylabel_text = m_data.columns[i][1]
        else:
            ylabel_text = ''

        if vertical_legend:
            axes = fig.add_subplot(grid_sp[i, 0])
        else:
            axes = fig.add_subplot(grid_sp[i])

        def format_fn(tick_val, tick_pos):
            """
            Map each tick with the corresponding
            elapsed time to label the timeline.
            """
            values_range = range(len(m_data_array[0]))

            if int(tick_val) in values_range:
                return time_deltas[int(tick_val)].seconds
            else:
                return ''

        axes.format(xticklabels=format_fn, ylocator=[0.5],
                    yticklabels=[ylabel_text], ticklabelsize='small')

        data = np.array(m_data_array[i], ndmin=2)

        # Generate the timeline gradient
        axes.imshow(data, cmap=ListedColormap(list(reversed(cc.bgy))),
                    norm=norm, aspect='auto',
                    vmin=norm.vmin, vmax=norm.vmax)

    if not vertical_legend:
        col_bar_ax = fig.add_subplot(grid_sp[-1], autoshare=False, ticklabelsize='small')
        fig.colorbar(cm.ScalarMappable(
            cmap=ListedColormap(list(reversed(cc.bgy))), norm=norm),
            cax=col_bar_ax, orientation="horizontal")
    else:
        col_bar_ax = fig.add_subplot(grid_sp[:, 1], ticklabelsize='small')
        fig.colorbar(cm.ScalarMappable(
            cmap=ListedColormap(list(reversed(cc.bgy))), norm=norm),
            cax=col_bar_ax)

    return fig


def agg_metric_timeline(df, metric, fig_fn, fig_title=''):
    """
    Create and save a figure timeline from the DataFrame `df`, which contains
    EAR loop data, for metric/s that match the regular expression `metric`.
    The resulting figure shows the aggregated value of the metric along all
    involved nodes in EAR loop data.
    """

    fig = generate_metric_timeline_fig(df, metric,
                                       fig_title=fig_title, granularity='app')
    fig.savefig(fig_fn)


def metric_timeline(df, metric, fig_fn, fig_title='', **kwargs):
    fig = generate_metric_timeline_fig(df, metric,
                                       fig_title=fig_title, **kwargs)
    fig.savefig(fig_fn)


def runtime(filename, avail_metrics, req_metrics, rel_range=False, save=False,
            title=None, job_id=None, step_id=None, output=None,
            horizontal_legend=False):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `avail_metrics` supported.
    """

    df = (read_data(filename, sep=';')
          .pipe(filter_df, JOBID=job_id, STEPID=step_id, JID=job_id)
          .pipe(filter_invalid_gpu_series)
          .pipe(df_gpu_node_metrics)
          )

    for metric in req_metrics:
        # Get a valid EAR column name
        # metric_name = avail_metrics.get_metric(metric).name
        metric_config = avail_metrics[metric]
        metric_name = metric_config['column_name']

        # Set the configured normalization if requested.
        norm = None
        if not rel_range:
            metric_range = metric_config['range']
            print(f"Metric range: {metric_range}")

            norm = Normalize(vmin=metric_range[0],
                             vmax=metric_range[1], clip=True)

        fig_title = metric
        if title:  # We preserve the title got by the user
            fig_title = f'{title}: {metric}'
        else:  # The default title: %metric-%job_id-%step_id
            if job_id:
                fig_title = '-'.join([fig_title, str(job_id)])
                if step_id is not None:
                    fig_title = '-'.join([fig_title, str(step_id)])

        fig = generate_metric_timeline_fig(df, metric_name, norm=norm,
                                           fig_title=fig_title,
                                           vertical_legend=not horizontal_legend)

        if save:
            name = f'runtime_{metric}'
            if job_id:
                name = '-'.join([name, str(job_id)])
                if step_id is not None:
                    name = '-'.join([name, str(step_id)])

            if output:
                if os.path.isdir(output):

                    name = os.path.join(output, name)
                else:
                    name = '-'.join([output, name])

            print(f'storing figure {name}')

            fig.savefig(name)
        else:
            fig.show()


def ear2prv(job_data_fn, loop_data_fn, events_data_fn=None, job_id=None,
            step_id=None, output_fn=None,
            events_config_fn=None):

    # Read the Job data

    df_job = (read_data(job_data_fn, sep=';')
              .pipe(filter_df, id=job_id, step_id=step_id)
              .pipe(lambda df: df[['id', 'step_id', 'app_id',
                                   'start_time', 'end_time']])
              .pipe(lambda df: df.rename(columns={"id": "JOBID",
                                                  "step_id": "STEPID",
                                                  'app_id': 'app_name'}))
              )
    # Read the Loop data

    df_loops = (read_data(loop_data_fn, sep=';')
                .pipe(filter_df, JOBID=job_id, STEPID=step_id)
                .merge(df_job)
                .assign(
                    # Paraver works with integers
                    CPI=lambda df: df.CPI * 1000000,
                    ITER_TIME_SEC=lambda df: df.ITER_TIME_SEC * 1000000,
                    IO_MBS=lambda df: df.IO_MBS * 1000000,
                    # Paraver works at microsecond granularity
                    time=lambda df: (df.TIMESTAMP -
                                     df.start_time) * 1000000
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
                # Drop unnecessary columns
                .drop(['LOOPID', 'LOOP_NEST_LEVEL', 'LOOP_SIZE',
                       'TIMESTAMP', 'start_time', 'end_time'], axis=1)
                )

    # Read EAR events data

    df_events = None

    if events_data_fn:
        cols_dict = {'JOBID': 'Job_id', 'STEPID': 'Step_id'}

        # By now events are in a space separated csv file.
        df_events = (read_data(events_data_fn, sep=r'\s+')
                     .pipe(filter_df, Job_id=job_id, Step_id=step_id)
                     .merge(df_job.rename(columns=cols_dict))
                     .assign(
                         # Paraver works at microsecond granularity
                         time=lambda df: (df.Timestamp -
                                          df.start_time) * 1000000,
                     )
                     .join(pd.Series(dtype=np.int64, name='task_id'))
                     .join(pd.Series(dtype=np.int64, name='app_id'))
                     .join(pd.Series(dtype=np.int64, name='event_type'))
                     # Drop unnecessary columns
                     .drop(['Event_ID', 'Timestamp',
                            'start_time', 'end_time'], axis=1)
                     )

    # ### Paraver trace header
    #
    # **Important note** By now this tool assumes all nodes
    # (where each one will be converted to a Paraver task) involved in a
    # job-step use the same number of GPUs (where each one will be converted to
    # a Paraver thread). It's needed to know how eacct command handles the
    # resulting header when there is a different number of GPUs for each
    # job-step requested.
    # [UPDATE: Now, eacct has a column for all EAR supported GPUs even that
    # GPUx has no data.]
    #
    # It is also assumed that both events and loops are from the same Job-step,
    # executed on the same node set..
    #
    # #### Generic info of the trace file

    node_info = np.sort(pd.unique(df_loops.NODENAME))
    n_nodes = 0

    if df_events is not None and not \
            np.array_equal(node_info, np.sort(pd.unique(df_events.node_id))):
        print('ERROR: Loops and events data do not have'
              f' the same node information: {node_info}, '
              f'{np.sort(pd.unique(df_events.node_id))}')
        return
    else:
        n_nodes = node_info.size

        f_time = (np.max(df_job.end_time) -
                  np.min(df_job.start_time)) * 1000000

    # #### Getting Application info
    #
    # An EAR Job-Step is a Paraver Application
    #
    # *By now this approach is useless, since we restrict the user to provide
    # only a Job-Step. But managing more than one Paraver application
    # (a Job-Step) is maintained to be more easy in a future if we wanted to
    # work with multiple applications at once.*

    appl_info = df_loops.groupby(['JOBID', 'STEPID']).groups
    n_appl = len(appl_info)

    # #### Generating the Application list and
    # Paraver's Names Configuration File (.row)
    #
    # EAR reports node metrics. Each EAR node is a Paraver task, so in the most
    # of cases a Paraver user can visualize the EAR data at least at the task
    # level. There is one exception where the user will visualize different
    # information when working at the Paraver's thread level: for GPU
    # information. If EAR data contains information about GPU(s) on a node, the
    # information of each GPU associated to that node can be visualized at the
    # thread level. In other words, a node (task) have one or more GPUs
    # (threads).

    # The application level names section (.row) can be created here
    appl_lvl_names = f'LEVEL APPL SIZE {n_appl}'

    # The task level names section can be created here.
    # WARNING Assuming that we are only dealing with one application
    task_lvl_names = f'LEVEL TASK SIZE {n_nodes * n_appl}'

    total_task_cnt = 0  # The total number of tasks

    appl_lists = []  # Application lists of all applications

    # Thread level names for .row file. Only used if data contain GPU info
    thread_lvl_names = []

    # We count here the total number of threads
    # (used later for the Names Configuration file).
    total_threads_cnt = 0

    for appl_idx, (app_job, app_step) in enumerate(appl_info):

        df_app = df_loops[(df_loops['JOBID'] == app_job) &
                          (df_loops['STEPID'] == app_step)]

        appl_nodes = np.sort(pd.unique(df_loops.NODENAME))

        if df_events is not None:
            # Used only to check whether data correspond to the same Job-Step
            df_events_app = df_events[(df_events['Job_id'] == app_job) &
                                      (df_events['Step_id'] == app_step)]
            if not np.array_equal(appl_nodes,
                                  np.sort(pd.unique(df_events_app.node_id))):
                print('ERROR: Loops and events data do not have'
                      ' the same node information.')
                return

        n_tasks = appl_nodes.size
        total_task_cnt += n_tasks

        # An EAR GPU is a Paraver thread
        gpu_info = df_app.filter(regex=r'GPU\d_POWER_W').columns
        n_threads = gpu_info.size

        # We accumulate the number of GPUs (paraver threads)
        total_threads_cnt += (n_threads * n_tasks)

        # print(f'{appl_idx + 1}) {app_job}-{app_step}: {n_tasks} '
        #       f'task(s), nodes {appl_nodes}, {n_threads} GPUs (threads)\n')

        # Create here the application list, and append to the global appl list
        appl_list = [f'{max(n_threads, 1)}:{node_idx + 1}'
                     for node_idx, _ in enumerate(appl_nodes)]
        appl_lists.append(f'{n_tasks}({",".join(appl_list)})')

        # Set each row its corresponding Appl Id
        df_loops.loc[(df_loops['JOBID'] == app_job) &
                     (df_loops['STEPID'] == app_step), 'app_id'] = \
            np.int64(appl_idx + 1)

        if df_events is not None:
            df_events.loc[(df_events['Job_id'] == app_job) &
                          (df_events['Step_id'] == app_step), 'app_id'] =\
                np.int64(appl_idx + 1)

        # TASK level names

        for node_idx, node_name in enumerate(appl_nodes):
            # Set each row its corresponding Task Id
            df_loops.loc[(df_loops['JOBID'] == app_job) &
                         (df_loops['STEPID'] == app_step) &
                         (df_loops['NODENAME'] == node_name), 'task_id'] \
                = np.int64(node_idx + 1)

            if df_events is not None:
                df_events.loc[(df_events['Job_id'] == app_job) &
                              (df_events['Step_id'] == app_step) &
                              (df_events['node_id'] == node_name), 'task_id'] \
                    = np.int64(node_idx + 1)

            task_lvl_names = '\n'.join([task_lvl_names, f' {node_name}'])

            # THREAD NAMES
            for gpu_idx in range(n_threads):
                (thread_lvl_names
                 .append(f'({node_name}) GPU {gpu_idx} @ {node_name}'))

        # APPL level names
        appl_lvl_names = '\n'.join([appl_lvl_names,
                                    f'({np.int64(appl_idx + 1)})'
                                    f' {df_app.app_name.unique()[0]}'])

    # The resulting Application List
    appl_list_str = ':'.join(appl_lists)

    names_conf_str = '\n'.join([appl_lvl_names, task_lvl_names])

    thread_lvl_names_str = ''
    if total_threads_cnt != 0:
        # Some application has GPUs, so we can configure and the THREAD level
        thread_lvl_names_str = '\n'.join(['LEVEL THREAD SIZE'
                                          f' {total_threads_cnt}',
                                          '\n'.join(thread_lvl_names)])

        names_conf_str = '\n'.join([names_conf_str, thread_lvl_names_str])

    # Store the Names Configuration File (.row)
    if not output_fn:
        output_fn = loop_data_fn.partition('.')[0]

    with open('.'.join([output_fn, 'row']), 'w') as row_file:
        row_file.write(names_conf_str)

    # #### Generating the Paraver trace header

    date_time = time.strftime('%d/%m/%y at %H:%M',
                              time.localtime(np.min(df_job.start_time)))

    file_trace_hdr = (f'#Paraver ({date_time}):{f_time}'
                      f':0:{n_appl}:{appl_list_str}')

    # ### Paraver trace body

    # #### Loops

    metrics = (df_loops.drop(columns=['JOBID', 'STEPID', 'NODENAME',
                                      'time', 'task_id', 'app_id', 'app_name',
                                      'gpu_power', 'gpu_freq', 'gpu_mem_freq',
                                      'gpu_util', 'gpu_mem_util']
                             ).columns
               )

    # We first sort data by timestamp in ascending
    # order as specified by Paraver trace format.
    trace_sorted_df = df_loops.sort_values('time')

    records = trace_sorted_df.to_records(index=False)
    columns = trace_sorted_df.columns

    app_id_idx = columns.get_loc('app_id')
    task_id_idx = columns.get_loc('task_id')
    timestamp_idx = columns.get_loc('time')

    gpu_field_regex = re.compile(r'GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                                 r'UTIL_PERC|MEM_UTIL_PERC)')
    gpu_field_map = {'POWER_W': 'gpu_power',
                     'FREQ_KHZ': 'gpu_freq',
                     'MEM_FREQ_KHZ': 'gpu_mem_freq',
                     'UTIL_PERC': 'gpu_util',
                     'MEM_UTIL_PERC': 'gpu_mem_util',
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
                thread_idx = int(gpu_field.group(1)) + 1

                metric_idx = columns.get_loc(gpu_field_map[gpu_field.group(2)])

            body_list.append(f'2:0:{"{:0.0f}".format(row[app_id_idx])}'
                             f':{"{:0.0f}".format(row[task_id_idx])}'
                             f':{thread_idx}:{row[timestamp_idx]}'
                             f':{metric_idx}:{event_val}')

    # #### Loops configuration file

    cols_regex = re.compile(r'(GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                            r'UTIL_PERC|MEM_UTIL_PERC))'
                            r'|JOBID|STEPID|NODENAME|LOOPID|LOOP_NEST_LEVEL|'
                            r'LOOP_SIZE|TIMESTAMP|start_time|end_time|time|'
                            r'task_id|app_id|app_name')
    metrics = (df_loops
               .drop(columns=df_loops.filter(regex=cols_regex).columns)
               .columns)

    # A map with metric_name metric_idx
    metric_event_typ_map = {metric: trace_sorted_df.columns.get_loc(metric)
                            for metric in metrics}

    event_typ_lst = [f'EVENT_TYPE\n0\t{metric_event_typ_map[metric]}'
                     f'\t{metric}\n' for metric in metric_event_typ_map]

    # #### EAR events body and configuration
    if df_events is not None:

        # The starting Event identifier for EAR events
        ear_events_id_off = max(metric_event_typ_map.values()) + 1

        # Get all EAR events types
        events_info = pd.unique(df_events.Event_type)

        for event_idx, event_t in enumerate(events_info):

            # We set the configuration of the EAR event type
            event_typ_str = (f'EVENT_TYPE\n0\t{event_idx + ear_events_id_off}'
                             f'\t{event_t}\n')
            event_typ_lst.append(event_typ_str)

            # Set the event identifier taking into account the offset made by
            # loops metrics identifiers
            df_events.loc[df_events['Event_type'] == event_t,
                          'event_type'] = event_idx + ear_events_id_off

        event_typ_lst.append('\n')

        # We get a sorted (by time) DataFrame
        df_events_sorted = (df_events
                            .astype(
                                {'task_id': int,
                                 'app_id': int,
                                 'event_type': int})
                            .sort_values('time'))

        smft = '2:0:{app_id}:{task_id}:1:{time}:{event_type}:{Value}'.format

        ear_events_body_list = (df_events_sorted
                                .apply(lambda x: smft(**x), axis=1)
                                .to_list())

        # #### Merging

        # We use the heapq merge function where the key is the
        # time field (position 5) of the trace row.
        file_trace_body = '\n'.join(merge(body_list,
                                          ear_events_body_list,
                                          key=lambda x: x.split(sep=':')[5])
                                    )
    else:
        file_trace_body = '\n'.join(body_list)

    with open('.'.join([output_fn, 'prv']), 'w') as prv_file:
        prv_file.write('\n'.join([file_trace_hdr, file_trace_body]))

    # ## Paraver Configuration File

    def_options_str = 'DEFAULT_OPTIONS\n\nLEVEL\tTASK\nUNITS\tSEC\n'

    # Merging default settings with event types
    paraver_conf_file_str = '\n'.join([def_options_str,
                                       '\n'.join(event_typ_lst)])

    # Adding the categorical labels for EAR events.
    if df_events is not None:

        # Set the default config filename if the user didn't give one
        if events_config_fn is None:
            events_config_fn = 'events_config.json'

        if (os.path.isfile(events_config_fn)):
            """
            # Hardcoded configuration - version 0:
            ear_event_types_values = {
                    'earl_state': {
                        0: 'NO_PERIOD',
                        1: 'FIRST_ITERATION',
                        2: 'EVALUATING_LOCAL_SIGNATURE',
                        3: 'SIGNATURE_STABLE',
                        4: 'PROJECTION_ERROR',
                        5: 'RECOMPUTING_N',
                        6: 'SIGNATURE_HAS_CHANGED',
                        7: 'TEST_LOOP',
                        8: 'EVALUATING_GLOBAL_SIGNATURE',
                    },
                    'policy_accuracy': {
                        0: 'OPT_NOT_READY',
                        1: 'OPT_OK',
                        2: 'OPT_NOT_OK',
                        3: 'OPT_TRY_AGAIN',
                    },
                    'earl_phase': {
                        1: 'APP_COMP_BOUND',
                        2: 'APP_MPI_BOUND',
                        3: 'APP_IO_BOUND',
                        4: 'APP_BUSY_WAITING',
                        5: 'APP_CPU_GPU',
                    },
                }
            """
            with open(events_config_fn, 'r', encoding='utf-8') as f:
                try:
                    ear_event_types_values = json.load(f)

                    for ear_event_type in ear_event_types_values:
                        idx = paraver_conf_file_str.find(ear_event_type)
                        if idx != -1:
                            values_str = ('\n'
                                          .join([f'{key}\t{value}'
                                                 for key, value
                                                 in (ear_event_types_values[ear_event_type]
                                                     .items()
                                                     )
                                                 ]
                                                )
                                          )

                            st_p = idx + len(ear_event_type)

                            paraver_conf_file_str = ('\n'
                                                     .join([paraver_conf_file_str[:st_p],
                                                            'VALUES',
                                                            values_str,
                                                            paraver_conf_file_str[st_p+1:]
                                                            ]
                                                           )
                                                     )
                except json.JSONDecodeError as json_err:
                    print(f'ERROR: Decoding {json_err.doc}\n'
                          f'Message: "{json_err.msg}" at line '
                          f'{json_err.lineno} column {json_err.colno}.')

        else:
            print('WARNING: Events configuration file '
                  f'{events_config_fn} does not exist.')
    else:
        print('There are not EAR events.')

    with open('.'.join([output_fn, 'pcf']), 'w') as pcf_file:
        pcf_file.write(paraver_conf_file_str)


def eacct(result_format, jobid, stepid, ear_events=False):
    # A temporary folder to store the generated csv file
    csv_file = '.'.join(['_'.join(['tmp', f"{jobid}", f"{stepid}"]), 'csv'])

    if result_format == "runtime":
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", "-c", csv_file]
    elif result_format == "ear2prv":
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", "-o", "-c", csv_file]
    elif result_format == 'job-summary':
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-l", "-c", csv_file]

    else:
        print("Unrecognized format: Please contact with support@eas4dc.com")
        sys.exit()

    # Run the command
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check the possible errors
    if "Error getting ear.conf path" in res.stderr.decode('utf-8'):
        print("Error getting ear.conf path")
        sys.exit()

    if "No jobs found" in res.stdout.decode('utf-8'):
        print(f"eacct: {jobid} No jobs found.")
        sys.exit()

    if "No loops retrieved" in res.stdout.decode('utf-8'):
        print(f"eacct: {jobid}.{stepid} No loops retrieved")
        sys.exit()

    # Request EAR events

    if ear_events or result_format == 'job-summary':
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-x", '-c',
               '.'.join(['events', csv_file])]
        res = subprocess.run(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    if result_format == 'job-summary':
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", '-c',
               '.'.join(['loops', csv_file])]
        res = subprocess.run(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    # Return generated file
    return csv_file


def parser_action(args):

    csv_generated = False

    if args.input_file is None:

        # Action performing eacct command and storing csv files

        input_file = eacct(args.format, args.job_id, args.step_id, args.events)

        args.input_file = input_file

        csv_generated = True

    if args.format == "runtime":

        runtime(args.input_file, read_metrics_configuration('config.json'),
                args.metrics, args.relative_range, args.save, args.title,
                args.job_id, args.step_id, args.output, args.horizontal_legend)

    elif args.format == "ear2prv":
        head_path, tail_path = os.path.split(args.input_file)
        out_jobs_path = os.path.join(head_path,
                                     '.'.join(['out_jobs', tail_path]))

        events_data_path = None
        if args.events:
            events_data_path = (os.path
                                .join(head_path,
                                      '.'.join(['events', tail_path])))

        # Call ear2prv format method
        ear2prv(out_jobs_path, args.input_file,
                events_data_fn=events_data_path, job_id=args.job_id,
                step_id=args.step_id, output_fn=args.output,
                events_config_fn=args.events_config)

    elif args.format == 'job-summary':
        df_long = (read_data(args.input_file, sep=';')
                   .pipe(filter_df,
                         JOBID=args.job_id,
                         STEPID=args.step_id))

        head_path, tail_path = os.path.split(args.input_file)

        df_loops_path = os.path.join(head_path,
                                     '.'.join(['loops', tail_path])
                                     )
        df_loops = (read_data(df_loops_path, sep=';')
                    .pipe(filter_df, JOBID=args.job_id, STEPID=args.step_id))

        df_events_path = os.path.join(head_path,
                                      '.'.join(['events', tail_path])
                                      )
        df_events = (read_data(df_events_path, sep=r'\s+')
                     .pipe(filter_df,
                           Job_id=args.job_id,
                           Step_id=args.step_id))

        metrics_conf = read_metrics_configuration('config.json')
        phases_conf = read_phases_configuration('config.json')

        job_summary(df_long, df_loops, df_events, metrics_conf, phases_conf)

    if csv_generated and not args.keep_csv:
        os.system(f'rm {input_file}')
        if args.format == 'ear2prv':
            os.system(f'rm {out_jobs_path}')
            if args.events:
                os.system(f'rm {events_data_path}')
        if args.format == 'job_summary':
            os.system(f'rm {df_loops_path} && rm {df_events_path}')


def build_parser():
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

    parser = argparse.ArgumentParser(description='''High level support for read
                                     and visualize EAR job data.''',
                                     formatter_class=formatter,
                                     epilog='Contact: support@eas4dc.com')
    parser.add_argument('--version', action='version', version='%(prog)s 4.2')

    parser.add_argument('--format', required=True,
                        choices=['runtime', 'ear2prv', 'job-summary'],
                        help='''Build results according to chosen format:
                        runtime (static images) or ear2prv (using paraver
                        tool).''')

    parser.add_argument('--input-file', help=('''Specifies the input file(s)
                                              name(s) to read data from.
                                              It can be a path.'''))

    parser.add_argument('-j', '--job-id', type=int, required=True,
                        help='Filter the data by the Job ID.')
    parser.add_argument('-s', '--step-id', type=int, required=True,
                        help='Filter the data by the Step ID.')

    # ONLY for runtime format
    runtime_group_args = parser.add_argument_group('`runtime` format options')

    group = runtime_group_args.add_mutually_exclusive_group()
    group.add_argument('--save', action='store_true',
                       help='Activate the flag to store resulting figures.')
    group.add_argument('--show', action='store_true',
                       help='Show the resulting figure (default).')

    runtime_group_args.add_argument('-t', '--title',
                                    help="""Set the resulting figure title.
                                    Only valid for `runtime` format option.
                                    The resulting title will be
                                    "<title>: <metric>" for each requested
                                    metric.""")

    runtime_group_args.add_argument('-r', '--relative-range',
                                    action='store_true',
                                    help='Use the relative range of a metric '
                                    'over the trace data to build the '
                                    'gradient, instead of the manually '
                                    'specified at config.ini file.')

    runtime_group_args.add_argument('-l', '--horizontal-legend',
                                    action='store_true',
                                    help='Display the legend horizontally. '
                                    'This option is useful when your trace has'
                                    ' a low number of nodes.')

    config_metrics = read_metrics_configuration('config.json')

    metrics_help_str = ('Space separated list of case sensitive'
                        ' metrics names to visualize. Allowed values are '
                        f'{", ".join(config_metrics)}'
                        )
    runtime_group_args.add_argument('-m', '--metrics',
                                    help=metrics_help_str,
                                    metavar='metric', nargs='+',
                                    choices=config_metrics.keys())

    ear2prv_group_args = parser.add_argument_group('`ear2prv` format options')

    events_help_str = 'Include EAR events in the trace fille.'
    ear2prv_group_args.add_argument('-e', '--events', action='store_true',
                                    help=events_help_str)

    events_config_help_str = ('Specify a (JSON formatted) file with event'
                              ' types categories. Default: events_config.json')
    ear2prv_group_args.add_argument('--events-config',
                                    help=events_config_help_str)

    parser.add_argument('-o', '--output',
                        help="""Sets the output file name.
                        If a path to an existing directory is given,
                        `runtime` option saves files with the form
                        `runtime_<metric>` (for each requested metric) will be
                        saved on the given directory. Otherwise,
                        <output>-runtime_<metric> is stored for each resulting
                        figure.
                        For ear2prv format, specify the base Paraver trace
                        files name.""")

    parser.add_argument('-k', '--keep-csv', action='store_true',
                        help='Don\'t remove temporary csv files.')

    # parser.set_defaults(func=parser_action_closure(conf_metrics))
    parser.set_defaults(func=parser_action)

    return parser


def main():
    """ Entry method. """

    parser = build_parser()

    args = parser.parse_args()

    # condition if input file not given
    args.func(args)

    # run query and plot generate phase plots
    return args


if __name__ == '__main__':
    args = main()
