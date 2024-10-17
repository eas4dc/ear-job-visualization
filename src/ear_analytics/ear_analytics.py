######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


""" High level support for read and visualize
    information given by EARL. """

import sys
from argparse import HelpFormatter, ArgumentParser
from os import path, system
import subprocess
from time import strftime, localtime
import re

import numpy as np

import pandas as pd
import heapq

from importlib_resources import files

from itertools import chain

from .metrics import read_metrics_configuration, get_plottable_metrics

from .utils import (filter_df, read_job_data_config, read_loop_data_config,
                    function_compose)

from . import ear_data as edata
from . import static_figures
from . import io_api

from .events import read_events_configuration


def metric_timeline(df, metric, step, fig_fn, fig_title='', **kwargs):
    fig = static_figures.generate_metric_timeline_fig(df, metric, step,
                                                      fig_title=fig_title,
                                                      **kwargs)
    fig.savefig(fig_fn)


def runtime(filename, out_jobs_fn, req_metrics, config_fn,
            rel_range=True, title=None, job_id=None, step_id=None,
            output=None):
    """
    This function generates a heatmap of runtime metrics requested by
    `req_metrics`.

    It also receives the `filename` to read data from,
    and `avail_metrics` supported.
    """
    avail_metrics = read_metrics_configuration(config_fn)
    try:
        df = (io_api.read_data(filename, sep=';')
              .pipe(filter_df, JOBID=job_id, STEPID=step_id, JID=job_id)
              .pipe(edata.filter_invalid_gpu_series, config_fn)
              .pipe(edata.df_gpu_node_metrics, config_fn)
              )
        df_job = (io_api.read_data(out_jobs_fn, sep=';')
                  .pipe(filter_df, JOBID=job_id, STEPID=step_id,
                        id=job_id, step_id=step_id))
    except FileNotFoundError as e:
        print(e)
        return
    else:
        # We need the application start time
        configuration = io_api.read_configuration(config_fn)

        start_time_col = configuration['columns']['app_info']['start_time']
        app_start_time = df_job[start_time_col].min()
        
        end_time_col = configuration['columns']['app_info']['end_time']
        app_end_time = df_job[end_time_col].max()

        for metric in req_metrics:
            # Get a valid EAR column name
            metric_config = avail_metrics[metric]

            metric_name = metric_config['column_name']
            dsply_nm = metric_config.get('display_name', metric_name)
            step = metric_config['step']

            # Set the configured normalization if requested.
            v_min = None
            v_max = None
            if not rel_range:
                metric_range = metric_config['range']
                print(f"Configured metric range: {metric_range}")
                v_min = metric_range[0]
                v_max = metric_range[1]

            # TODO: Add the min/max value of the metric (relative range always)
            fig_title = metric
            if title:  # We preserve the title got by the user
                fig_title = f'{title}: {metric}'
            else:  # The default title: %metric-%job_id-%step_id
                if job_id:
                    fig_title = '-'.join([fig_title, str(job_id)])
                    if step_id is not None:
                        fig_title = '-'.join([fig_title, str(step_id)])

            gpu_metrics_re = configuration['columns']['gpu_data']['gpu_columns_re']
            fig = (static_figures
                   .generate_metric_timeline_fig(df, app_start_time,
                                                 app_end_time,
                                                 metric_name, step,
                                                 v_min=v_min,
                                                 v_max=v_max,
                                                 fig_title=fig_title,
                                                 metric_display_name=dsply_nm,
                                                 gpu_metrics_re=gpu_metrics_re))

            # if save:
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

            fig.savefig(name, dpi='figure', bbox_inches='tight')
            # else:
            #     fig.show()


def ear2prv(job_data_fn, loop_data_fn, job_data_config, loop_data_config,
            events_config, events_data_fn=None, job_id=None, step_id=None,
            output_fn=None, events_config_fn=None):

    def filter_df_columns(df, cols_config):
        """
        Filters df based on column names configured in cols_config
        """
        regex = '|'.join(cols_config.keys())
        return df.filter(regex=regex)

    def set_df_types(df, cols_config):
        """
        Returns df with types set by cols_config
        """
        ret_df = pd.DataFrame(index=df.index)
        dfs = [(df
                .filter(regex=regex)
                .pipe(lambda df: df.astype(cols_config[regex])
                      if not df.empty else df)
                ) for regex in cols_config.keys()]

        return ret_df.join(dfs)

    def insert_initial_values(df_loops, df_job):
        """
        This function inserts a row on df_loops for each unique
        job, step, node tuple, with all values to 0 except TIMESTAMP,
        got from start_time of the corresponding job, step in df_job.
        """

        task_fields = ['JOBID', 'STEPID', 'APPID', 'NODENAME']
        group_by_task = df_loops.groupby(task_fields).groups

        jobs = []
        steps = []
        apps = []
        nodes = []
        times = []
        for j, s, a, n in group_by_task:

            task_start_time = df_job.loc[(df_job['JOBID'] == j) &
                                         (df_job['STEPID'] == s) &
                                         (df_job['APPID'] == a)]['START_TIME']

            if not task_start_time.empty:
                jobs += [j]
                steps += [s]
                apps += [a]
                nodes += [n]
                times += [task_start_time.iat[0]]  # There is a unique element
            else:
                print(f"Warning! Job data hasn't information about job {j} "
                      f"step {s} app {a}. This job-step-app won't be on the "
                      "output trace.")

        df_start_time = (pd.DataFrame({'JOBID': jobs, 'STEPID': steps,
                                       'APPID': apps, 'NODENAME': nodes,
                                       'TIMESTAMP': times},
                                      columns=df_loops.columns)
                         .fillna(0))

        return pd.concat([df_loops, df_start_time], ignore_index=True)

    def multiply_floats_by_1000000(df):
        df_floats = (df.select_dtypes(include=['Float64'])
                     .apply(lambda x: x*1000000)
                     .astype('Int64'))
        df_non_float = df.select_dtypes(exclude=['Float64'])
        return df_floats.join(df_non_float)

    def insert_jobdata(df_loops, df_job):
        return df_loops.merge(df_job[['JOBID', 'STEPID', 'APPID',
                                      'JOBNAME', 'START_TIME', 'END_TIME']])

    def print_df(df):
        """
        Utility function to be used in a pipe
        """
        print(df)
        return df

    # Read the Job data

    df_job = (read_data(job_data_fn, sep=';')
              .pipe(filter_df, JOBID=job_id, id=job_id,
                    STEPID=step_id, step_id=step_id)
              .pipe(filter_df_columns, job_data_config)
              .pipe(set_df_types, job_data_config)
              )

    # Read the Loop data
    df_loops = (read_data(loop_data_fn, sep=';')
                .pipe(filter_df, JOBID=job_id, STEPID=step_id)
                .pipe(filter_df_columns, loop_data_config)
                .pipe(set_df_types, loop_data_config)
                .pipe(insert_initial_values, df_job)
                .assign(
                    # Paraver works at microsecond granularity
                    time=lambda df: (df.TIMESTAMP -
                                     df_job.START_TIME.min()) * 1000000
                    )
                .pipe(multiply_floats_by_1000000)
                .pipe(insert_jobdata, df_job)
                .join(pd.Series(dtype='Int64', name='task_id'))
                .join(pd.Series(dtype='Int64', name='app_id'))
                .join(pd.Series(dtype='Int64', name='gpu_power'))
                .join(pd.Series(dtype='Int64', name='gpu_freq'))
                .join(pd.Series(dtype='Int64', name='gpu_mem_freq'))
                .join(pd.Series(dtype='Int64', name='gpu_util'))
                .join(pd.Series(dtype='Int64', name='gpu_mem_util'))
                .join(pd.Series(dtype='Int64', name='gpu_gflops'))
                .join(pd.Series(dtype='Int64', name='dcgm_gr_engine_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_sm_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_sm_occupancy'))
                .join(pd.Series(dtype='Int64', name='dcgm_pipe_tensor_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_pipe_fp64_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_pipe_fp32_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_pipe_fp16_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_dram_active'))
                .join(pd.Series(dtype='Int64', name='dcgm_nvlink_tx_bytes'))
                .join(pd.Series(dtype='Int64', name='dcgm_nvlink_rx_bytes'))
                .join(pd.Series(dtype='Int64', name='dcgm_pcie_tx_bytes'))
                .join(pd.Series(dtype='Int64', name='dcgm_pcie_rx_bytes'))
                )
    # print(df_loops.info())

    # Read EAR events data

    df_events = None

    # TODO: time must be computed based on the start of the batch job
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
                     .join(pd.Series(dtype='Int64', name='task_id'))
                     .join(pd.Series(dtype='Int64', name='app_id'))
                     .join(pd.Series(dtype='Int64', name='event_type'))
                     # Drop unnecessary columns
                     .drop(['Event_ID', 'Timestamp',
                            'start_time', 'end_time'], axis=1)
                     )
    # else:
        # print("No events file provided.")

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

        f_time = (df_job.END_TIME.max() -
                  df_job.START_TIME.min()) * 1000000

        print(f'Number of nodes: {n_nodes}. Total trace duration: {f_time}')

    # #### Getting Application info
    #
    # An EAR Job-Step-App is a Paraver Application
    #

    appl_info = df_loops.groupby(['JOBID', 'STEPID', 'APPID']).groups
    n_appl = len(appl_info)

    print(f'Number of applications (job-step): {n_appl}')

    # #### Generating the Application list and
    # Paraver's Names Configuration File (.row)
    #
    # EAR reports node metrics. Each EAR node is a Paraver task, so in most
    # cases a Paraver user can visualize the EAR data at least at the task
    # level. There is one exception where the user will visualize different
    # information when working at the Paraver's thread level: for GPU
    # information. If EAR data contains information about GPU(s) on a node, the
    # information of each GPU associated to that node can be visualized at the
    # thread level. In other words, a node (task) have one or more GPUs
    # (threads).

    # The application level names section (.row) can be created here
    appl_lvl_names = f'LEVEL APPL SIZE {n_appl}'

    # The task level names section
    task_lvl_names = ''

    total_task_cnt = 0  # The total number of tasks

    appl_lists = []  # Application lists of all applications

    # Thread level names for .row file. Only used if data contain GPU info
    thread_lvl_names = []

    # We count here the total number of threads
    # (used later for the Names Configuration file).
    total_threads_cnt = 0

    for appl_idx, (app_job, app_step, app_appid) in enumerate(appl_info):

        df_app = df_loops[(df_loops['JOBID'] == app_job) &
                          (df_loops['STEPID'] == app_step) &
                          (df_loops['APPID'] == app_appid)]

        appl_nodes = np.sort(pd.unique(df_app.NODENAME))

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
        total_task_cnt += n_tasks  # Task count used after the for loop

        # An EAR GPU is a Paraver thread
        gpu_info = df_app.filter(regex=r'GPU\d_POWER_W').columns
        n_threads = gpu_info.size

        # We accumulate the number of GPUs (paraver threads)
        total_threads_cnt += (n_threads * n_tasks)

        # print(f'{appl_idx + 1}) {app_job}-{app_step}: {n_tasks} '
        #       f'task(s) (nodes {appl_nodes}), {n_threads} GPUs (threads)\n')

        # Create here the application list, and append to the global appl list
        appl_list = [f'{max(n_threads, 1)}:{node_idx + 1}'
                     for node_idx, _ in enumerate(appl_nodes)]
        appl_lists.append(f'{n_tasks}({",".join(appl_list)})')

        # Set each row its corresponding Appl Id
        df_loops.loc[(df_loops['JOBID'] == app_job) &
                     (df_loops['STEPID'] == app_step) &
                     (df_loops['APPID'] == app_appid), 'app_id'] = \
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
                         (df_loops['APPID'] == app_appid) &
                         (df_loops['NODENAME'] == node_name), 'task_id'] \
                = np.int64(node_idx + 1)

            if df_events is not None:
                df_events.loc[(df_events['Job_id'] == app_job) &
                              (df_events['Step_id'] == app_step) &
                              (df_events['node_id'] == node_name), 'task_id'] \
                    = np.int64(node_idx + 1)

            task_fmt = f'({app_job}.{app_step}.{app_appid}) @ {node_name}'
            task_lvl_names = '\n'.join([task_lvl_names, task_fmt])

            # THREAD NAMES
            for gpu_idx in range(n_threads):
                thread_fmt = (f'({app_job}.{app_step}.{app_appid}) '
                              f'GPU {gpu_idx} @ {node_name}')
                thread_lvl_names.append(thread_fmt)

        # APPL level names
        appl_lvl_names = '\n'.join([appl_lvl_names,
                                    f'({app_job}.{app_step}.{app_appid})'
                                    f' {df_app.JOBNAME.unique()[0]}'])

    task_lvl_names = ''.join([f'LEVEL TASK SIZE {total_task_cnt}',
                              task_lvl_names])

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
                         localtime(np.min(df_job.START_TIME)))

    file_trace_hdr = (f'#Paraver ({date_time}):{f_time}'
                      f':0:{n_appl}:{appl_list_str}')

    # ### Paraver trace body

    # #### Loops

    metrics = (df_loops.drop(columns=['JOBID', 'STEPID', 'APPID', 'NODENAME',
                                      'time', 'task_id', 'app_id', 'JOBNAME',
                                      'gpu_power', 'gpu_freq', 'gpu_mem_freq',
                                      'gpu_util', 'gpu_mem_util', 'gpu_gflops',
                                      'dcgm_gr_engine_active',
                                      'dcgm_sm_active', 'dcgm_sm_occupancy',
                                      'dcgm_pipe_tensor_active',
                                      'dcgm_pipe_fp64_active',
                                      'dcgm_pipe_fp32_active',
                                      'dcgm_pipe_fp16_active',
                                      'dcgm_dram_active',
                                      'dcgm_nvlink_tx_bytes',
                                      'dcgm_nvlink_rx_bytes',
                                      'dcgm_pcie_tx_bytes',
                                      'dcgm_pcie_rx_bytes', 'TIMESTAMP',
                                      'START_TIME', 'END_TIME']
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
                                 r'UTIL_PERC|MEM_UTIL_PERC|GFLOPS|'
                                 r'gr_engine_active|sm_active|sm_occupancy|'
                                 r'tensor_active|fp64_active|fp32_active|'
                                 r'fp16_active|dram_active|nvlink_tx_bytes|'
                                 r'nvlink_rx_bytes|pcie_tx_bytes|'
                                 r'pcie_rx_bytes)')
    gpu_field_map = {'POWER_W': 'gpu_power',
                     'FREQ_KHZ': 'gpu_freq',
                     'MEM_FREQ_KHZ': 'gpu_mem_freq',
                     'UTIL_PERC': 'gpu_util',
                     'MEM_UTIL_PERC': 'gpu_mem_util',
                     'GFLOPS': 'gpu_gflops',
                     'gr_engine_active': 'dcgm_gr_engine_active',
                     'sm_active': 'dcgm_sm_active',
                     'sm_occupancy': 'dcgm_sm_occupancy',
                     'tensor_active': 'dcgm_pipe_tensor_active',
                     'fp64_active': 'dcgm_pipe_fp64_active',
                     'fp32_active': 'dcgm_pipe_fp32_active',
                     'fp16_active': 'dcgm_pipe_fp16_active',
                     'dram_active': 'dcgm_dram_active',
                     'nvlink_tx_bytes': 'dcgm_nvlink_tx_bytes',
                     'nvlink_rx_bytes': 'dcgm_nvlink_rx_bytes',
                     'pcie_tx_bytes': 'dcgm_pcie_tx_bytes',
                     'pcie_rx_bytes': 'dcgm_pcie_rx_bytes'
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
                            r'UTIL_PERC|MEM_UTIL_PERC|GFLOPS|gr_engine_active|'
                            r'sm_active|sm_occupancy|tensor_active|'
                            r'fp64_active|fp32_active|fp16_active|dram_active|'
                            r'nvlink_tx_bytes|nvlink_rx_bytes|pcie_tx_bytes|'
                            r'pcie_rx_bytes))|JOBID|STEPID|NODENAME|LOOPID|'
                            r'LOOP_NEST_LEVEL|LOOP_SIZE|TIMESTAMP|START_TIME|'
                            r'END_TIME|time|task_id|app_id|JOBNAME|APPID')
    metrics = (df_loops
               .drop(columns=df_loops.filter(regex=cols_regex).columns)
               .columns)

    # A map with metric_name metric_idx
    metric_event_typ_map = {metric: trace_sorted_df.columns.get_loc(metric)
                            for metric in metrics}

    event_typ_lst = [f'EVENT_TYPE\n0\t{metric_event_typ_map[metric]}'
                     f'\t{metric}\n' for metric in metric_event_typ_map]

    # States body and configuration
    df_states = (df_loops
                 .groupby(['app_id', 'task_id'])[['START_TIME', 'END_TIME']].max()
                 .assign(state_id=1,  # 1 -> Running
                         START_TIME=lambda df: (df.START_TIME - df_job.START_TIME.min()) * 1000000,
                         END_TIME=lambda df: (df.END_TIME - df_job.START_TIME.min()) * 1000000)
                 .reset_index())

    smft = '1:0:{app_id}:{task_id}:1:{START_TIME}:{END_TIME}:{state_id}'.format
    states_body_list = (df_states
                        .apply(lambda x: smft(**x), axis=1)
                        .to_list()
                        )

    # Start time and end time events
    start_end_event_ids = {event: columns.get_loc(event)
                           for event in ['START_TIME', 'END_TIME']}

    df_start_end_time = (df_loops
                         .groupby(['app_id', 'task_id'])[['START_TIME', 'END_TIME']].max()
                         .reset_index()
                         .melt(id_vars=['app_id', 'task_id'])
                         .assign(
                             event_id=lambda df: df.variable.map(lambda x: start_end_event_ids[x]),
                             time=lambda df: (df.value - df_job.START_TIME.min()) * 1000000
                             )
                         .drop(columns='variable')
                         )

    smft = '2:0:{app_id}:{task_id}:1:{time}:{event_id}:{value}'.format
    start_end_body_list = (df_start_end_time
                           .apply(lambda x: smft(**x), axis=1)
                           .to_list()
                           )

    start_end_event_types = [f'EVENT_TYPE\n0\t{start_end_event_ids[event]}'
                             f'\t{event}\n' for event in start_end_event_ids]
    event_typ_lst += start_end_event_types

    def sort_by_record_type(trace_list):
        """
        Descending order
        """
        return sorted(trace_list, key=lambda x: int(x.split(sep=':')[0]),
                      reverse=True)

    def sort_by_timestamp(trace_list):
        """
        Ascending order
        """
        return sorted(trace_list, key=lambda x: int(x.split(sep=':')[5]))

    # first we worder by record type as it is the second sorting criteria and
    # we make use of sorted() stable property:
    # https://docs.python.org/3/howto/sorting.html#sort-stability-and-complex-sorts
    sort_by_type_and_time = function_compose(sort_by_timestamp,
                                             sort_by_record_type)

    body_list_sorted = sort_by_type_and_time(chain(states_body_list, body_list,
                                                   start_end_body_list))

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
        file_trace_body = '\n'.join(heapq.merge(body_list_sorted,
                                    ear_events_body_list,
                                    key=lambda x: x.split(sep=':')[5])
                                    )
    else:
        file_trace_body = '\n'.join(body_list_sorted)

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
    # else:
    #     print('There are not EAR events.')

    with open('.'.join([output_fn, 'pcf']), 'w') as pcf_file:
        pcf_file.write(paraver_conf_file_str)


def eacct(result_format, jobid, stepid=None, ear_events=False):
    """
    This function calls properly the `eacct` command in order
    to get files to be worked by `result_format` feature.

    The filename where data is stored is "tmp_<jobid>[_<stepid>].csv", which is
    returned as str. '_<stepid>' region depends on whether `stepid` parameter
    is not None.

    Basic command for each format:
        runtime -> -r -o -> Generates [out_jobs.]tmp_<jobid>[_<stepid>].csv
        ear2prv -> -r -o -> Generates [out_jobs.]tmp_<jobid>[_<stepid>].csv
        summary -> -l -> Generates [events.]tmp_<jobid>[_<stepid>].csv

    If the requested format is "summary" or `ear_events` is True, an
    additional call is done requesting for events, i.e., `eacct -x`.
    The resulting filename is "events.tmp_<jobid>[_<stepid>].csv", but note
    that the function is still returning the basic command filename.
    """

    if stepid is None:
        csv_loops_file = f'tmp_{jobid}_loops.csv'
        csv_apps_file = f'tmp_{jobid}_apps.csv'
        job_fmt = f'{jobid}'
    else:
        csv_loops_file = f'tmp_{jobid}_{stepid}_loops.csv'
        csv_apps_file = f'tmp_{jobid}_{stepid}_apps.csv'
        job_fmt = f'{jobid}.{stepid}'

    if result_format == 'runtime' or result_format == "ear2prv":
        cmd_loops = ["eacct", "-j", job_fmt, "-r", "-c", csv_loops_file]
        cmd_apps = ["eacct", "-j", job_fmt, "-l", "-c", csv_apps_file]

        # Run the command
        try:
            res_loops = subprocess.run(cmd_loops, capture_output=True,
                                       check=True)
            res_apps = subprocess.run(cmd_apps, capture_output=True,
                                      check=True)
        except subprocess.CalledProcessError as check_failed:
            sys.exit(f'Command `{check_failed.cmd}` returned an error: '
                     f'{check_failed.stderr.decode("utf-8")}')
        except OSError as os_error:
            sys.exit(f'OS returned an error: ({os_error.errno})'
                     f' {os_error.strerror}: "{os_error.filename}"')
        else:
            print(f'{res_loops.args} ran successfully:'
                  f'\n{res_loops.stdout.decode("utf-8")}')

            print(f'{res_apps.args} ran successfully:'
                  f'\n{res_apps.stdout.decode("utf-8")}')

            # Return generated file
            return csv_loops_file, csv_apps_file
    else:
        sys.exit(f'Unrecognized format: {result_format}')


def parser_action(args):
    """
    Parses the Namespace `args` and decides which action to do.
    """

    # Get the (possible) config file provided by the user
    if args.config_file:
        config_file_path = args.config_file
    else:
        config_file_path = files('ear_analytics').joinpath('config.json')

    # Print configuration file
    if args.print_config == True:
        io_api.print_configuration(config_file_path)
        sys.exit()

    print(f'Using {config_file_path} as configuration file...')

    # Show available metrics
    if args.avail_metrics is True:

        comp = function_compose(get_plottable_metrics,
                                read_metrics_configuration)
        config_metrics = comp(config_file_path)
        print(f'Available metrics: {" ".join(config_metrics)}.')
        sys.exit()

    csv_generated = False

    if args.loops_file is None:
        # sys.exit('This version still requires an input file.'
        #          ' Run an applicatin with --ear-user-db flag.')

        # Action performing eacct command and storing csv files

        args.loops_file, args.apps_file = eacct(args.format, args.job_id,
                                                args.step_id)

        csv_generated = True

    if args.format == "runtime":

        runtime(args.loops_file, args.apps_file,
                args.metrics, config_file_path, args.manual_range,
                args.title, args.job_id, args.step_id, args.output)

    elif args.format == "ear2prv":
        events_data_path = None

        # Call ear2prv format method
        ear2prv(args.apps_file, args.loops_file,
                read_job_data_config(config_file_path),
                read_loop_data_config(config_file_path),
                read_events_configuration(config_file_path),
                events_data_fn=events_data_path, job_id=args.job_id,
                step_id=args.step_id, output_fn=args.output)

    if csv_generated and not args.keep_csv:
        system(f'rm {args.loops_file}')
        system(f'rm {args.apps_file}')


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
    parser.add_argument('--version', action='version', version='%(prog)s 5.1')

    main_group = parser.add_argument_group('Main options',
                                           description='''The main option flags
                                           required by the tool.''')

    main_group.add_argument('-c', '--config-file',
                            help='Specify a custom configuration file.')

    # format and print-config options are mutually exclusive
    main_excl_grp = main_group.add_mutually_exclusive_group(required=True)

    # Specify
    main_excl_grp.add_argument('--format', choices=['runtime', 'ear2prv'],
                               help='''Build results according to chosen format:
                               `runtime` (static images) or `ear2prv` (using
                                paraver tool).''')

    main_excl_grp.add_argument('--print-config', action='store_true',
                               help='''Prints the used configuration file.''')

    main_excl_grp.add_argument('--avail-metrics', action='store_true',
                               help='''Prints the available metrics provided by
                               the configuration file.''')

    format_grp = parser.add_argument_group('Format common options',
                                           description='''Used when requesting
                                           any of "--format" choices.''')

    format_grp.add_argument('--loops-file', required='--apps-file' in sys.argv,
                            help='''Specifies the input file(s)
                             name(s) to read data from. It can be a path.''')

    format_grp.add_argument('--apps-file', required='--loops-file' in sys.argv,
                            help='''Specifies the input file(s) name(s) to
                             read data from. It can be a path.''')

    format_grp.add_argument('-j', '--job-id', type=int,
                            help='Filter the data by the Job ID.',
                            required='--format' in sys.argv)

    format_grp.add_argument('-s', '--step-id', type=int,
                            help='Filter the data by the Step ID.')

    format_grp.add_argument('-o', '--output',
                            help="""Sets the output file name.
                            If a path to an existing directory is given,
                            `runtime` option saves files with the form
                            `runtime_<metric>.pdf` (for each requested metric)
                             will be on the given directory. Otherwise,
                            runtime_<metric>-<output> is stored for each
                             resulting figure.
                            For ear2prv format, specify the base Paraver trace
                            files base name.""")

    format_grp.add_argument('-k', '--keep-csv', action='store_true',
                            help='Don\'t remove temporary csv files.')

    # ONLY for runtime format
    runtime_grp = parser.add_argument_group('`runtime` format options',
                                            description='''Used when
                                            requesting "--format runtime".''')

    runtime_grp.add_argument('-t', '--title',
                             help="""Set the resulting figure title.
                             The resulting title will be
                             "<title>: <metric>" for each requested
                             metric.""")

    runtime_grp.add_argument('-r', '--manual-range',
                             action='store_false',
                             help='''Uses the range of values specified in the
                             configuration file to build the final trace
                             colormap insted of building it based on the
                             range of the data source's metric.''')

    metrics_help_str = ('Space separated list of case sensitive'
                        ' metrics names to visualize. Allowed values can '
                        'be viewed with `ear-job-analytics --avail-metrics`.')
    runtime_grp.add_argument('-m', '--metrics', help=metrics_help_str,
                             metavar='metric', nargs='+')

    return parser


def main():
    """ Entry method. """

    parser = build_parser()

    args = parser.parse_args()

    parser_action(args)

    sys.exit()


if __name__ == '__main__':
    main()
