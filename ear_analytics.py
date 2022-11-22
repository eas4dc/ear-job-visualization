""" High level support for read and visualize
    information given by EARL. """

import argparse
import os
import sys
import subprocess
import time
import re
import json

from heapq import merge

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
from common.utils import filter_df, list_str


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

    for m in args.metrics:
        if m not in list(mtrcs.metrics.keys()):
            print("error: argument -m/--metrics: invalid choice: ", m)
            print("choose from:", list(mtrcs.metrics.keys()))
            return

    df = (read_data(filename)
          .pipe(filter_df, JOBID=job_id, STEPID=step_id, JID=job_id)
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


def ear2prv(job_data_fn, loop_data_fn, events_data_fn=None, job_id=None,
            step_id=None, output_fn=None,
            events_config_fn=None):

    # Read the Job data

    df_job = (read_data(job_data_fn)
              .pipe(filter_df, id=job_id, step_id=step_id)
              .pipe(lambda df: df[['id', 'step_id', 'app_id',
                                   'start_time', 'end_time']])
              .pipe(lambda df: df.rename(columns={"id": "JOBID",
                                                  "step_id": "STEPID",
                                                  'app_id': 'app_name'}))
              )
    # Read the Loop data

    df_loops = (read_data(loop_data_fn)
                .pipe(filter_df, JOBID=job_id, STEPID=step_id)
                .merge(df_job)
                .assign(
                    # Paraver works with integers
                    CPI=lambda df: df.CPI * 1000000,
                    TIME=lambda df: df.TIME * 1000000,  # ear ITER_TIME_SEC
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
                    {'TIME': np.int64,  # ear4.2 ITER_TIME_SEC
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
                .drop(['FIRST_EVENT', 'LEVEL', 'SIZE',
                       'TIMESTAMP', 'start_time', 'end_time'], axis=1)
                )

    # Read EAR events data

    df_events = None

    if events_data_fn:
        # By now events are in a space separated csv file.
        df_events = (read_data(events_data_fn, sep=r'\s+')
                     .pipe(filter_df, Job_id=job_id, Step_id=step_id)
                     .merge(df_job.rename(columns={'JOBID': 'Job_id', 'STEPID': 'Step_id'}))
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
              f' the same node information: {node_info}, {np.sort(pd.unique(df_events.node_id))}')
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
        gpu_info = df_app.filter(regex='GPOWER').columns
        n_threads = gpu_info.size

        # We accumulate the number of GPUs (paraver threads)
        total_threads_cnt += n_threads

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

    gpu_field_regex = re.compile(r'(GPOWER|GFREQ|GMEMFREQ|GUTIL|GMEMUTIL)(\d)')
    gpu_field_map = {'GPOWER': 'gpu_power',
                     'GFREQ': 'gpu_freq',
                     'GMEMFREQ': 'gpu_mem_freq',
                     'GUTIL': 'gpu_util',
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

    # #### Loops configuration file

    cols_regex = re.compile(r'((GPOWER|GFREQ|GMEMFREQ|GUTIL|GMEMUTIL)(\d))'
                            r'|JOBID|STEPID|NODENAME|FIRST_EVENT|LEVEL|SIZE'
                            r'|TIMESTAMP|start_time|end_time|time|task_id'
                            r'|app_id|app_name')
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

    with open('.'.join([output_fn, 'pcf']), 'w') as pcf_file:
        pcf_file.write(paraver_conf_file_str)


def eacct(result_format, jobid, stepid, ear_events=False):
    # A temporary folder to store the generated csv file
    csv_file = '.'.join(['_'.join(['tmp', f"{jobid}", f"{stepid}"]), 'csv'])

    if result_format == "runtime":
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", "-c", csv_file]
    elif result_format == "ear2prv":
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-r", "-o", "-c", csv_file]
    else:
        print("Unrecognized format: Please contact with support@eas4dc.com")
        sys.exit()

    # Run the command
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check the possible errors
    if "Error getting ear.conf path" in res.stderr.decode('utf-8') :
        print("Error getting ear.conf path")
        sys.exit()

    if "No jobs found" in res.stdout.decode('utf-8') :
        print(f"eacct: {jobid} No jobs found.")
        sys.exit()

    if "No loops retrieved" in res.stdout.decode('utf-8') :
        print(f"eacct: {jobid}.{stepid} No loops retrieved")
        sys.exit()

    # Request EAR events
    if ear_events:
        cmd = ["eacct", "-j", f"{jobid}.{stepid}", "-x", '-c', '.'.join(['events', csv_file])]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Return generated file
    return csv_file


def parser_action_closure(conf_metrics):

    def parser_action(args):

        csv_generated = False

        if args.input_file is None:
            # Action performing eacct command and storing csv files
            input_file = eacct(args.format, args.jobid, args.stepid, args.events)
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

            events_data_path = None
            if args.events:
                events_data_path = (os.path
                                    .join(head_path,
                                          '.'.join(['events', tail_path])))

            # Call ear2prv format method
            ear2prv(out_jobs_path, args.input_file,
                    events_data_fn=events_data_path, job_id=args.jobid,
                    step_id=args.stepid, output_fn=args.output,
                    events_config_fn=args.events_config)

        if csv_generated and not args.keep_csv:
            os.system(f'rm {input_file}')
            if args.format == 'ear2prv':
                os.system(f'rm {out_jobs_path}')
                if args.events:
                    os.system(f'rm {events_data_path}')

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
    parser.add_argument('--version', action='version', version='%(prog)s 4.1')

    parser.add_argument('--format', required=True,
                        choices=['runtime', 'ear2prv'],
                        help='Build results according to chosen format: '
                        'runtime (static images) or ear2prv (using paraver '
                        'tool).')

    parser.add_argument('--input-file', help=('Specifies the input file(s) '
                                              'name(s) to read data from.'))

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
                                    help='Set the resulting figure title '
                                    '(Only valid with runtime format).')

    runtime_group_args.add_argument('-r', '--relative-range',
                                    action='store_true',
                                    help='Use the relative range of a metric '
                                    'over the trace data to build the '
                                    'gradient.')

    runtime_group_args.add_argument('-l', '--horizontal-legend',
                                    action='store_true',
                                    help='Display the legend horizontally. '
                                    'This option is useful when your trace has'
                                    ' a low number of nodes.')

    runtime_group_args.add_argument('-n', '--node',
                                    help=('Filter the data by the node '
                                          '(used ONLY for phase visualisation).')
                                    )

    metrics_help_str = ('Space separated list of case sensitive'
                        ' metrics names to visualize. Allowed values are '
                        f'{", ".join(conf_metrics.metrics.keys())}'
                        )
    runtime_group_args.add_argument('-m', '--metrics', type=list_str,
                                    default=['cpi', 'gbs', 'gflops'],
                                    help=metrics_help_str, metavar='metric')

    ear2prv_group_args = parser.add_argument_group('`ear2prv` format options')

    events_help_str = 'Include EAR events in the trace fille.'
    ear2prv_group_args.add_argument('-e', '--events', action='store_true',
                                    help=events_help_str)

    events_config_help_str = ('Specify a (JSON formatted) file with event'
                              ' types categories. Default: events_config.json')
    ear2prv_group_args.add_argument('--events-config', help=events_config_help_str)

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

    # run query and plot generate phase plots
    return args


if __name__ == '__main__':
    args = main()
