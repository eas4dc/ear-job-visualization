""" High level support for read and visualize
    information given by EARL. """

import sys
from argparse import HelpFormatter, ArgumentParser
from os import mkdir, path, system
from subprocess import run, PIPE, STDOUT, CalledProcessError
from time import strftime, localtime
import re
import json

import pandas as pd
import numpy as np
from pylatex import Command

from heapq import merge

from proplot import figure, GridSpec
from matplotlib import cm
from matplotlib.colors import Normalize

from importlib_resources import files

from .io_api import read_data
from .metrics import read_metrics_configuration, metric_regex
from .utils import filter_df

from . import ear_data as edata

from .phases import (read_phases_configuration,
                     df_phases_phase_time_ratio,
                     df_phases_to_tex_tabular)

from .job_summary import (job_cpu_summary_df,
                          job_summary_to_tex_tabular,
                          job_gpu_summary,
                          job_gpu_summary_to_tex_tabular)

from .events import read_events_configuration


def build_job_summary(df_long, df_loops, df_phases, metrics_conf, phases_conf):
    """
    Generate a job summary.
    """
    print('Building job summary...')

    job_id = df_long['JOBID'].unique()
    if job_id.size != 1:
        print('ERROR: Only one job is supported. Jobs detected: {job_id}.')
        return
    else:
        job_id = str(job_id[0])

    try:
        print(f'Creating directory for job {job_id}...')
        mkdir(job_id)
    except FileExistsError:
        print(f'Error: Directory {job_id} already exists.')
        return

    try:

        print('Getting main file from template...')

        main_file_path = path.join(job_id, 'main.tex')
        main_file_template = files('ear_analytics').joinpath('templates/main.tex.template')

        cmd = ' '.join(['cp', str(main_file_template), main_file_path])

        run(cmd, stdout=PIPE, stderr=STDOUT, check=True, shell=True)

    except CalledProcessError as err:
        print('Error copying the template tex file:',
              err.returncode, f'({err.output})')
        return

    # Build the resulting document title.
    text_dir = path.join(job_id, 'text')

    print(f'Creating {text_dir} directory...')
    mkdir(text_dir)

    job_name = df_long['JOBNAME'].unique()[0]
    title = Command('title', f'{job_name} report')
    title.generate_tex(path.join(text_dir, 'title'))

    tables_dir = path.join(job_id, 'tables')

    print(f'Creating {tables_dir} directory...')
    mkdir(tables_dir)

    # Job summary

    print('Building job summary table')

    job_sum_fn = path.join(tables_dir, 'job_summary')

    (df_long
     .pipe(job_cpu_summary_df, metrics_conf)
     .pipe(job_summary_to_tex_tabular, job_sum_fn)
     )

    # Job summary (GPU part)

    gpu_sum_file_path = path.join(text_dir, 'job_gpu_summary.tex')

    if (edata.df_has_gpu_data(df_long)):

        try:

            print('Getting job GPU summary file from template...')

            gpu_sum_file_template = files('ear_analytics').joinpath('templates/text/job_gpu_summary.tex')

            cmd = ' '.join(['cp', str(gpu_sum_file_template), gpu_sum_file_path])

            run(cmd, stdout=PIPE, stderr=STDOUT, check=True, shell=True)

        except CalledProcessError as err:
            print('Error copying the template tex file:',
                  err.returncode, f'({err.output})')
            return

        job_gpusum_fn = path.join(tables_dir, 'job_gpu_summary')

        # Build the DataFrame with summary and create the tabular
        (df_long
         .pipe(job_gpu_summary, metrics_conf, job_gpusum_fn)
         )
    else:
        # Create an empry file
        try:
            with open(gpu_sum_file_path, mode='w'):
                pass
        except OSError:
            print('Error: Creating the dummy GPU summary table file.')
            return

    # Phases summary

    print('Building phases summary...')

    job_phasesum_fn = path.join(tables_dir, 'job_phases_summary')

    (df_phases
     .pivot(index='node_id', columns='Event_type', values='Value')
     .pipe(df_phases_phase_time_ratio, phases_conf)
     .pipe(df_phases_to_tex_tabular, job_phasesum_fn)
     )

    timelines_dir = path.join(job_id, 'timelines')

    print('Creating timelines dir...')
    mkdir(timelines_dir)

    print('Building job timelines...')

    # Aggregated power
    agg_metric_timeline(df_loops, metric_regex('dc_power', metrics_conf),
                        path.join(timelines_dir, 'agg_dcpower'),
                        fig_title='Accumulated DC Node Power (W)')

    # Aggregated Mem. bandwidth
    agg_metric_timeline(df_loops, metric_regex('gbs', metrics_conf),
                        path.join(timelines_dir, 'agg_gbs'),
                        fig_title='Accumulated memory bandwidth (GB/s)')

    # Aggregated GFlop/s
    agg_metric_timeline(df_loops, metric_regex('gflops', metrics_conf),
                        path.join(timelines_dir, 'agg_gflops'),
                        fig_title='Accumulated CPU GFlop/s')

    # Aggregated I/O
    agg_metric_timeline(df_loops, metric_regex('io_mbs', metrics_conf),
                        path.join(timelines_dir, 'agg_iombs'),
                        fig_title='Accumulated I/O throughput (MB/s)')

    # GPU timelines

    gpu_aggpwr_file_path = path.join(text_dir, 'agg_gpupwr.tex')
    gpu_util_file_path = path.join(text_dir, 'gpu_util.tex')

    if edata.df_has_gpu_data(df_loops):
        # Aggregated GPU power
        try:
            print('Getting job GPU agg power file from template...')

            gpu_aggpwr_file_template = files('ear_analytics').joinpath('templates/text/agg_gpupwr.tex')

            cmd = ' '.join(['cp', str(gpu_aggpwr_file_template), gpu_aggpwr_file_path])

            run(cmd, stdout=PIPE, stderr=STDOUT, check=True, shell=True)

        except CalledProcessError as err:
            print('Error copying the template tex file:',
                  err.returncode, f'({err.output})')
            return
        else:
            gpu_pwr_re = metric_regex('gpu_power', metrics_conf)

            df_agg_gpupwr = (df_loops
                             .assign(
                                 tot_gpu_pwr=lambda x: (x.filter(regex=gpu_pwr_re)
                                                         .sum(axis=1)
                                                        )
                                 )
                             )
            agg_metric_timeline(df_agg_gpupwr, 'tot_gpu_pwr',
                                path.join(timelines_dir, 'agg_gpupower'),
                                fig_title='Aggregated GPU Power (W)')
        # Per-node GPU util
        try:
            print('Getting job GPU util file from template...')

            gpu_util_file_template = files('ear_analytics').joinpath('templates/text/gpu_util.tex')

            cmd = ' '.join(['cp', str(gpu_util_file_template), gpu_sum_file_path])

            run(cmd, stdout=PIPE, stderr=STDOUT, check=True, shell=True)

        except CalledProcessError as err:
            print('Error copying the template tex file:',
                  err.returncode, f'({err.output})')
            return
        else:
            norm = Normalize(vmin=0, vmax=100, clip=True)
            metric_timeline(edata.filter_invalid_gpu_series(df_loops),
                            metric_regex('gpu_util', metrics_conf),
                            path.join(timelines_dir, 'per-node_gpuutil'),
                            norm=norm, fig_title='GPU utilization (%)')
    else:
        # Create an empty file
        try:
            with open(gpu_aggpwr_file_path, mode='w'):
                pass
        except OSError:
            print('Error: Creating the dummy GPU agg power file.')
            return
        # Create an empty file
        try:
            with open(gpu_util_file_path, mode='w'):
                pass
        except OSError:
            print('Error: Creating the dummy GPU util file.')
            return

    # Per-node CPI
    metric_timeline(df_loops, metric_regex('cpi', metrics_conf),
                    path.join(timelines_dir, 'per-node_cpi'),
                    fig_title='Cycles per Instruction')

    # Per-node GBS
    metric_timeline(df_loops, metric_regex('gbs', metrics_conf),
                    path.join(timelines_dir, 'per-node_gbs'),
                    fig_title='Memory bandwidth (GB/s)')

    # Per-node GFlop/s
    metric_timeline(df_loops, metric_regex('gflops', metrics_conf),
                    path.join(timelines_dir, 'per-node_gflops'),
                    fig_title='CPU GFlop/s')

    # Per-node Avg. CPU freq.
    metric_timeline(df_loops, metric_regex('avg_cpufreq', metrics_conf),
                    path.join(timelines_dir, 'per-node_avgcpufreq'),
                    fig_title='Avg. CPU frequency (kHz)')

    # Per-node DC Power
    metric_timeline(df_loops, metric_regex('dc_power', metrics_conf),
                    path.join(timelines_dir, 'per-node_dcpower'),
                    fig_title='DC node power (W)')


def generate_metric_timeline_fig(df, metric, norm=None, fig_title='',
                                 vertical_legend=False, granularity='node'):
    """
    Generates the timeline gradient.

    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """

    metric_filter = df.filter(regex=metric).columns

    if granularity != 'app':
        m_data = edata.metric_timeseries_by_node(df, metric_filter)
    else:
        m_data = edata.metric_agg_timeseries(df, metric_filter)

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

    fig = figure(sharey=False, refaspect=20,
                 suptitle=fig_title, suptitle_kw={'size': 'x-small'})

    if vertical_legend:
        grid_sp = GridSpec(nrows=len(m_data_array), ncols=2,
                           width_ratios=(0.95, 0.05), hspace=0)
    else:
        def metric_row(i):

            # returns whether row i corresponds to a metric timeline.

            return i < len(m_data_array)

        height_ratios = [0.8 / len(m_data_array)
                         if metric_row(i) else 0.2
                         for i in range(len(m_data_array) + 1)]

        hspaces = [0 if metric_row(i + 1) else None
                   for i in range(len(m_data_array))]

        grid_sp = GridSpec(nrows=len(m_data_array) + 1, ncols=1,
                           hratios=height_ratios,
                           hspace=hspaces)

    # grid_sp = GridSpec(nrows=len(m_data_array), ncols=1, hspace=0)

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
                    yticklabels=[ylabel_text], ticklabelsize='xx-small')

        data = np.array(m_data_array[i], ndmin=2)

        # Generate the timeline gradient
        axes.imshow(data, cmap='imola_r',
                    norm='linear', aspect='auto', discrete=False,
                    vmin=norm.vmin, vmax=norm.vmax)

    if not vertical_legend:
        col_bar_ax = fig.add_subplot(grid_sp[-1], autoshare=False,
                                     ticklabelsize='xx-small')

        fig.colorbar(cm.ScalarMappable(
            cmap='imola_r', norm=norm),
            orientation="horizontal", loc='b', cax=col_bar_ax)
    else:
        col_bar_ax = fig.add_subplot(grid_sp[:, 1], ticklabelsize='xx-small')
        fig.colorbar(cm.ScalarMappable(
            cmap='imola_r', norm=norm),
            loc='r', cax=col_bar_ax)

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


def runtime(filename, avail_metrics, req_metrics, config_fn, rel_range=False, save=False,
            title=None, job_id=None, step_id=None, output=None,
            horizontal_legend=False):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `avail_metrics` supported.
    """

    try:
        df = (read_data(filename, sep=';')
              .pipe(filter_df, JOBID=job_id, STEPID=step_id, JID=job_id)
              .pipe(edata.filter_invalid_gpu_series)
              .pipe(edata.df_gpu_node_metrics, config_fn)
              )
    except FileNotFoundError as e:
        print(e)
        return
    else:
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

            vertical_legend = not horizontal_legend

            fig = generate_metric_timeline_fig(df, metric_name, norm=norm,
                                               fig_title=fig_title,
                                               vertical_legend=vertical_legend)

            if save:
                name = f'runtime_{metric}'
                """
                if job_id:
                    name = '-'.join([name, str(job_id)])
                    if step_id is not None:
                        name = '-'.join([name, str(step_id)])
                """

                if output:
                    if path.isdir(output):

                        name = path.join(output, name)
                    else:
                        name = '-'.join([name, output])

                print(f'storing figure {name}')

                fig.savefig(name)
            else:
                fig.show()


def ear2prv(job_data_fn, loop_data_fn, events_config, events_data_fn=None, job_id=None,
            step_id=None, output_fn=None, events_config_fn=None):

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
                 .append(f'GPU {gpu_idx} @ {node_name}'))

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

    date_time = strftime('%d/%m/%y at %H:%M',
                         localtime(np.min(df_job.start_time)))

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
        ear_event_types_values = events_config

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
    else:
        print('There are not EAR events.')

    with open('.'.join([output_fn, 'pcf']), 'w') as pcf_file:
        pcf_file.write(paraver_conf_file_str)


def eacct(result_format, jobid, stepid, ear_events=False):
    """
    This function calls properly the `eacct` command in order
    to get files to be worked by `result_format` feature.

    The filename where data is stored is "tmp_<jobid>_<stepid>.csv", which is
    returned.

    Basic command for each format:
        runtime -> -r
        ear2prv -> -r -o
        summary -> -l

    If the requested format is "summary" or `ear_events` is True, an
    additional call is done requesting for events, i.e., `eacct -x`.

    The resulting filename is "events.tmp_<jobid>_<stepid>.csv", but note that
    the function still returning the basic command filename.
    """

    csv_file = f'tmp_{jobid}_{stepid}.csv'

    if result_format == 'runtime':
        cmd = ['eacct', '-j', f'{jobid}.{stepid}', '-r', '-c', csv_file]
    elif result_format == "ear2prv":
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", "-o", "-c", csv_file]
    elif result_format == 'summary':
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-l", "-c", csv_file]

    else:
        print("Unrecognized format: Please contact with support@eas4dc.com")
        sys.exit()

    # Run the command
    res = run(cmd, stdout=PIPE, stderr=PIPE)

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

    if ear_events or result_format == 'summary':
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-x", '-c',
               '.'.join(['events', csv_file])]
        res = run(cmd, stdout=PIPE, stderr=PIPE)

    if result_format == 'summary':
        output_fn = '.'.join(['loops', csv_file])
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", '-c', output_fn]
        res = run(cmd, stdout=PIPE, stderr=PIPE)

    # Return generated file
    return csv_file


def parser_action(args):
    """
    Parses the Namespace `args` and decides which action to do.
    """

    csv_generated = False

    if args.config_file:
        config_file_path = args.config_file
    else:
        config_file_path = files('ear_analytics').joinpath('config.json')

    if args.input_file is None:

        # Action performing eacct command and storing csv files

        input_file = eacct(args.format, args.job_id, args.step_id, args.events)

        args.input_file = input_file

        csv_generated = True

    if args.format == "runtime":

        runtime(args.input_file, read_metrics_configuration(config_file_path),
                args.metrics, config_file_path, args.relative_range, args.save, args.title,
                args.job_id, args.step_id, args.output, args.horizontal_legend)

    elif args.format == "ear2prv":
        head_path, tail_path = path.split(args.input_file)
        out_jobs_path = path.join(head_path,
                                  '.'.join(['out_jobs', tail_path]))

        events_data_path = None
        if args.events:
            events_data_path = (path
                                .join(head_path,
                                      '.'.join(['events', tail_path])))

        # Call ear2prv format method
        ear2prv(out_jobs_path, args.input_file, read_events_configuration(config_file_path),
                events_data_fn=events_data_path, job_id=args.job_id,
                step_id=args.step_id, output_fn=args.output)

    elif args.format == 'summary':
        try:
            df_long = (read_data(args.input_file, sep=';')
                       .pipe(filter_df,
                             JOBID=args.job_id,
                             STEPID=args.step_id))
        except FileNotFoundError:
            return
        else:
            head_path, tail_path = path.split(args.input_file)

            df_loops_path = path.join(head_path,
                                      '.'.join(['loops', tail_path])
                                      )
            try:
                df_loops = (read_data(df_loops_path, sep=';')
                            .pipe(filter_df,
                                  JOBID=args.job_id,
                                  STEPID=args.step_id
                                  )
                            )
            except FileNotFoundError:
                return
            else:
                df_events_path = path.join(head_path,
                                           '.'.join(['events', tail_path])
                                           )
                try:
                    df_events = (read_data(df_events_path, sep=r'\s+')
                                 .pipe(filter_df,
                                       Job_id=args.job_id,
                                       Step_id=args.step_id))
                except FileNotFoundError:
                    return
                else:
                    config_file = files('ear_analytics').joinpath('config.json')
                    metrics_conf = read_metrics_configuration(config_file)
                    phases_conf = read_phases_configuration(config_file)

                    build_job_summary(df_long, df_loops, df_events,
                                      metrics_conf, phases_conf)

    if csv_generated and not args.keep_csv:
        system(f'rm {input_file}')
        if args.format == 'ear2prv':
            system(f'rm {out_jobs_path}')
            if args.events:
                system(f'rm {events_data_path}')
        if args.format == 'summary':
            system(f'rm {df_loops_path} && rm {df_events_path}')


def build_parser():
    """
    Returns the parser to read and check command line arguments.
    """

    class CustomHelpFormatter(HelpFormatter):
        """
        This class was created in order to change the width of the
        help message of the parser. It's a bit tricky to use this, as
        HelpFormatter is not officialy documented.
        """
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

    parser = ArgumentParser(description='''High level support for read
                            and visualize EAR job data.''',
                            formatter_class=formatter,
                            epilog='Contact: support@eas4dc.com')
    parser.add_argument('--version', action='version', version='%(prog)s 4.2')

    parser.add_argument('--format', required=True,
                        choices=['runtime', 'ear2prv', 'summary'],
                        help='''Build results according to chosen format:
                        runtime (static images) or ear2prv (using paraver
                        tool) (ear2prv UNSTABLE).''')

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

    config = files('ear_analytics').joinpath('config.json')
    config_metrics = read_metrics_configuration(config)

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

    """
    events_config_help_str = ('Specify a (JSON formatted) file with event'
                              ' types categories. Default: events_config.json')
    ear2prv_group_args.add_argument('--events-config',
                                    help=events_config_help_str)
    """

    parser.add_argument('-o', '--output',
                        help="""Sets the output file name.
                        If a path to an existing directory is given,
                        `runtime` option saves files with the form
                        `runtime_<metric>.pdf` (for each requested metric) will be
                        on the given directory. Otherwise,
                        runtime_<metric>-<output> is stored for each resulting
                        figure.
                        For ear2prv format, specify the base Paraver trace
                        files base name.""")

    parser.add_argument('-k', '--keep-csv', action='store_true',
                        help='Don\'t remove temporary csv files.')

    parser.add_argument('-c', '--config-file',
                        help='Specify a custom configuration file.')

    return parser


def main():
    """ Entry method. """

    parser = build_parser()

    args = parser.parse_args()

    parser_action(args)

    return args


if __name__ == '__main__':
    args = main()
