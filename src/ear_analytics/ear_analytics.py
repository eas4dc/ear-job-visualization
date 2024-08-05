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
from os import mkdir, path, system
from subprocess import run, PIPE, STDOUT, CalledProcessError
from time import strftime, localtime
import re

import numpy as np

from pandas import to_datetime, date_range, Series, unique, DataFrame, concat
from pylatex import Command

from heapq import merge

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid

from importlib_resources import files

from itertools import chain

from .io_api import read_data, print_configuration

from .metrics import (metric_regex, metric_step, read_metrics_configuration,
                      get_plottable_metrics)

from .utils import (filter_df, read_job_data_config, read_loop_data_config,
                    function_compose)

from . import ear_data as edata

from .phases import (read_phases_configuration,
                     df_phases_phase_time_ratio,
                     df_phases_to_tex_tabular)

from .job_summary import (job_cpu_summary_df,
                          job_summary_to_tex_tabular,
                          job_gpu_summary,
                          )

from .events import read_events_configuration


def build_job_summary(df_long, df_loops, df_phases, metrics_conf, phases_conf):
    """
    Generate a job summary.
    """
    print('Building job summary...')

    job_id = df_long['JOBID'].unique()
    if job_id.size != 1:
        print(f'ERROR: Only one job is supported. Jobs detected: {job_id}.')
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

        main_file_template = (files('ear_analytics')
                              .joinpath('templates/main.tex.template'))

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

            templ_gpu = 'templates/text/job_gpu_summary.tex'
            gpu_sum_file_template = files('ear_analytics').joinpath(templ_gpu)

            cmd = ' '.join(['cp',
                            str(gpu_sum_file_template),
                            gpu_sum_file_path])

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
                        metric_step('dc_power', metrics_conf),
                        path.join(timelines_dir, 'agg_dcpower'),
                        fig_title='Accumulated DC Node Power (W)')

    # Aggregated Mem. bandwidth
    agg_metric_timeline(df_loops, metric_regex('gbs', metrics_conf),
                        metric_step('gbs', metrics_conf),
                        path.join(timelines_dir, 'agg_gbs'),
                        fig_title='Accumulated memory bandwidth (GB/s)')

    # Aggregated GFlop/s
    agg_metric_timeline(df_loops, metric_regex('gflops', metrics_conf),
                        metric_step('gflops', metrics_conf),
                        path.join(timelines_dir, 'agg_gflops'),
                        fig_title='Accumulated CPU GFlop/s')

    # Aggregated I/O
    agg_metric_timeline(df_loops, metric_regex('io_mbs', metrics_conf),
                        metric_step('io_mbs', metrics_conf),
                        path.join(timelines_dir, 'agg_iombs'),
                        fig_title='Accumulated I/O throughput (MB/s)')

    # GPU timelines

    gpu_aggpwr_file_path = path.join(text_dir, 'agg_gpupwr.tex')
    gpu_util_file_path = path.join(text_dir, 'gpu_util.tex')

    if edata.df_has_gpu_data(df_loops):
        # Aggregated GPU power
        try:
            print('Getting job GPU agg power file from template...')

            agg_gpu_templ = 'templates/text/agg_gpupwr.tex'
            gpu_aggpwr_file_template = (files('ear_analytics')
                                        .joinpath(agg_gpu_templ))

            cmd = ' '.join(['cp',
                            str(gpu_aggpwr_file_template),
                            gpu_aggpwr_file_path])

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
                                                        .sum(axis=1))
                                 )
                             )
            agg_metric_timeline(df_agg_gpupwr, 'tot_gpu_pwr',
                                metric_step('tot_gpu_pwr', metrics_conf),
                                path.join(timelines_dir, 'agg_gpupower'),
                                fig_title='Aggregated GPU Power (W)')
        # Per-node GPU util
        try:
            print('Getting job GPU util file from template...')

            gpu_util_file_template = (files('ear_analytics')
                                      .joinpath('templates/text/gpu_util.tex'))

            cmd = ' '.join(['cp', str(gpu_util_file_template), gpu_util_file_path])

            run(cmd, stdout=PIPE, stderr=STDOUT, check=True, shell=True)

        except CalledProcessError as err:
            print('Error copying the template tex file:',
                  err.returncode, f'({err.output})')
            return
        else:
            norm = mpl.colors.Normalize(vmin=0, vmax=100, clip=True)
            metric_timeline(edata.filter_invalid_gpu_series(df_loops),
                            metric_regex('gpu_util', metrics_conf),
                            metric_step('gpu_util', metrics_conf),
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
                    metric_step('cpi', metrics_conf),
                    path.join(timelines_dir, 'per-node_cpi'),
                    fig_title='Cycles per Instruction')

    # Per-node GBS
    metric_timeline(df_loops, metric_regex('gbs', metrics_conf),
                    metric_step('gbs', metrics_conf),
                    path.join(timelines_dir, 'per-node_gbs'),
                    fig_title='Memory bandwidth (GB/s)')

    # Per-node GFlop/s
    metric_timeline(df_loops, metric_regex('gflops', metrics_conf),
                    metric_step('gflops', metrics_conf),
                    path.join(timelines_dir, 'per-node_gflops'),
                    fig_title='CPU GFlop/s')

    # Per-node Avg. CPU freq.
    metric_timeline(df_loops, metric_regex('avg_cpufreq', metrics_conf),
                    metric_step('avg_cpufreq', metrics_conf),
                    path.join(timelines_dir, 'per-node_avgcpufreq'),
                    fig_title='Avg. CPU frequency (kHz)')

    # Per-node DC Power
    metric_timeline(df_loops, metric_regex('dc_power', metrics_conf),
                    metric_step('dc_power', metrics_conf),
                    path.join(timelines_dir, 'per-node_dcpower'),
                    fig_title='DC node power (W)')


def generate_metric_timeline_fig(df, app_start_time, app_end_time, metric,
                                 step, v_min=None, v_max=None, fig_title='',
                                 granularity='node', metric_display_name=''):
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

    m_data.index = to_datetime(m_data.index, unit='s')

    new_idx = (date_range(start=to_datetime(app_start_time, unit='s'),
                          end=to_datetime(app_end_time, unit='s'), freq='1s')
               .union(m_data.index))

    m_data = m_data.reindex(new_idx).bfill()

    m_data_array = m_data.values.transpose()

    if granularity == 'app':
        m_data_array = m_data_array.reshape(1, m_data_array.shape[0])

    # Compute time deltas to be showed to
    # the end user instead of an absolute timestamp.
    time_deltas = [i - m_data.index[0] for i in m_data.index]
    # print(time_deltas, len(time_deltas))

    # Create the resulting figure for current metric
    print("Creating the figure...")

    # fig, axs = plt.subplots(nrows=len(m_data_array), sharex=True,
    #                         squeeze=False, gridspec_kw={'hspace': 0},
    #                         layout='constrained', # height_ratios=height_ratios,
    #                         figsize=(6.4, 1 + (6.4/15) * len(m_data_array))
    #                         )
    # fig.get_layout_engine().set(h_pad=0, hspace=0)
    # fig = plt.figure(figsize=(6.4, (6.4/15) * len(m_data_array)))
    fig = plt.figure()
    # grid = ImageGrid(fig, 111, nrows_ncols=(len(m_data_array), 1),
    #                  axes_pad=0, label_mode='1', cbar_location='bottom',
    #                  cbar_mode='edge', share_all=True, cbar_pad='50%', cbar_size='33%')
    axs = ImageGrid(fig, 111, nrows_ncols=(len(m_data_array), 1),
                    axes_pad=0, label_mode='L', cbar_location='bottom',
                    cbar_mode='single', cbar_pad=0.5, cbar_size=0.3)

    print(f'Setting title: {fig_title}')
    axs[0].set_title(fig_title)
    # axs[0, 0].set_title(fig_title)

    # Normalize values

    cmap = mpl.colormaps['viridis_r']
    bounds = np.arange(v_min if v_min is not None else np.nanmin(m_data_array),
                       v_max + step if v_max is not None
                       else np.nanmax(m_data_array) + step,
                       step)
    if bounds.size > 16 or bounds.size < 5:
        print(f'Warning! {bounds.size} discrete intervals generated.')

        if bounds.size > 16:
            print(f'Consider increasing the step (currently {step})'
                  f' for {metric}.')
        else:
            print(f'Consider decreasing the step (currently {step})'
                  f' for {metric}.')

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    gpu_metric_regex_str = (r'GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                            r'UTIL_PERC|MEM_UTIL_PERC|'
                            r'(10[01][0-9]))')
    gpu_metric_regex = re.compile(gpu_metric_regex_str)

    for ax, (i, _) in zip(axs, enumerate(m_data_array)):
        if granularity != 'app':
            gpu_metric_match = gpu_metric_regex.search(m_data.columns[i][0])

            if gpu_metric_match:
                ylabel_text = (f'GPU{gpu_metric_match.group(1)}'
                               f' @ {m_data.columns[i][1]}')
            else:
                ylabel_text = m_data.columns[i][1]
        else:
            ylabel_text = ''

        ax.grid(axis='x', alpha=0.5)
        ax.set_aspect(1/30)

        ax.set_yticks([0], labels=[ylabel_text])
        data = np.array(m_data_array[i], ndmin=2)

        # Generate the timeline gradient
        im = ax.imshow(data, norm=norm, cmap=cmap,
                       interpolation='none', aspect=2*len(m_data_array))

        # if i < len(m_data_array) - 1:
        #     ax.tick_params(axis='x', which='both', bottom=False)

    def format_fn(tick_val):
        """
        Map each tick with the corresponding
        elapsed time to label the timeline.
        """
        return time_deltas[tick_val].seconds

    xticks = np.arange(len(m_data_array[0]), step=20)
    xticklabels = map(format_fn, xticks)

    axs[-1].set(xticks=xticks, xticklabels=xticklabels)
    axs[-1].minorticks_on()

    # Create the figure colorbar

    label = metric if metric_display_name == '' else metric_display_name
    axs.cbar_axes[0].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                              label=label, format=None)

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
    #              cax=cb[0], location='bottom', label=label, format='%.2f')

    return fig


def agg_metric_timeline(df, metric, step, fig_fn, fig_title=''):
    """
    Create and save a figure timeline from the DataFrame `df`, which contains
    EAR loop data, for metric/s that match the regular expression `metric`.
    The resulting figure shows the aggregated value of the metric along all
    involved nodes in EAR loop data.
    """

    fig = generate_metric_timeline_fig(df, metric, step, fig_title=fig_title,
                                       granularity='app')
    fig.savefig(fig_fn)


def metric_timeline(df, metric, step, fig_fn, fig_title='', **kwargs):
    fig = generate_metric_timeline_fig(df, metric, step, fig_title=fig_title,
                                       **kwargs)
    fig.savefig(fig_fn)


def runtime(filename, out_jobs_fn, avail_metrics, req_metrics, config_fn,
            rel_range=True, title=None, job_id=None, step_id=None,
            output=None):
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
        df_job = (read_data(out_jobs_fn, sep=';')
                  .pipe(filter_df, JOBID=job_id, STEPID=step_id,
                        id=job_id, step_id=step_id))
    except FileNotFoundError as e:
        print(e)
        return
    else:
        # We need the application start time
        app_start_time = df_job.START_TIME.min()
        app_end_time = df_job.END_TIME.max()

        for metric in req_metrics:
            # Get a valid EAR column name
            metric_config = avail_metrics[metric]

            metric_name = metric_config['column_name']
            disply_name = metric_config.get('display_name', metric_name)
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

            fig = generate_metric_timeline_fig(df, app_start_time,
                                               app_end_time, metric_name, step,
                                               v_min=v_min, v_max=v_max,
                                               fig_title=fig_title,
                                               metric_display_name=disply_name)

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
        ret_df = DataFrame(index=df.index)
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

        df_start_time = (DataFrame({'JOBID': jobs, 'STEPID': steps,
                                    'APPID': apps, 'NODENAME': nodes,
                                    'TIMESTAMP': times},
                                   columns=df_loops.columns)
                         .fillna(0))

        return concat([df_loops, df_start_time], ignore_index=True)

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
                .join(Series(dtype='Int64', name='task_id'))
                .join(Series(dtype='Int64', name='app_id'))
                .join(Series(dtype='Int64', name='gpu_power'))
                .join(Series(dtype='Int64', name='gpu_freq'))
                .join(Series(dtype='Int64', name='gpu_mem_freq'))
                .join(Series(dtype='Int64', name='gpu_util'))
                .join(Series(dtype='Int64', name='gpu_mem_util'))
                .join(Series(dtype='Int64', name='gpu_gflops'))
                .join(Series(dtype='Int64', name='dcgm_gr_engine_active'))
                .join(Series(dtype='Int64', name='dcgm_sm_active'))
                .join(Series(dtype='Int64', name='dcgm_sm_occupancy'))
                .join(Series(dtype='Int64', name='dcgm_pipe_tensor_active'))
                .join(Series(dtype='Int64', name='dcgm_pipe_fp64_active'))
                .join(Series(dtype='Int64', name='dcgm_pipe_fp32_active'))
                .join(Series(dtype='Int64', name='dcgm_pipe_fp16_active'))
                .join(Series(dtype='Int64', name='dcgm_dram_active'))
                .join(Series(dtype='Int64', name='dcgm_nvlink_tx_bytes'))
                .join(Series(dtype='Int64', name='dcgm_nvlink_rx_bytes'))
                .join(Series(dtype='Int64', name='dcgm_pcie_tx_bytes'))
                .join(Series(dtype='Int64', name='dcgm_pcie_rx_bytes'))
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
                     .join(Series(dtype='Int64', name='task_id'))
                     .join(Series(dtype='Int64', name='app_id'))
                     .join(Series(dtype='Int64', name='event_type'))
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

    node_info = np.sort(unique(df_loops.NODENAME))
    n_nodes = 0

    if df_events is not None and not \
            np.array_equal(node_info, np.sort(unique(df_events.node_id))):
        print('ERROR: Loops and events data do not have'
              f' the same node information: {node_info}, '
              f'{np.sort(unique(df_events.node_id))}')
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

        appl_nodes = np.sort(unique(df_app.NODENAME))

        if df_events is not None:
            # Used only to check whether data correspond to the same Job-Step
            df_events_app = df_events[(df_events['Job_id'] == app_job) &
                                      (df_events['Step_id'] == app_step)]
            if not np.array_equal(appl_nodes,
                                  np.sort(unique(df_events_app.node_id))):
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

            task_lvl_names = '\n'.join([task_lvl_names,
                                        f'({app_job}.{app_step}.{app_appid}) @ {node_name}'])

            # THREAD NAMES
            for gpu_idx in range(n_threads):
                (thread_lvl_names
                 .append(f'({app_job}.{app_step}.{app_appid}) GPU {gpu_idx} @ {node_name}'))

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
                                      'dcgm_gr_engine_active', 'dcgm_sm_active',
                                      'dcgm_sm_occupancy', 'dcgm_pipe_tensor_active',
                                      'dcgm_pipe_fp64_active', 'dcgm_pipe_fp32_active',
                                      'dcgm_pipe_fp16_active', 'dcgm_dram_active',
                                      'dcgm_nvlink_tx_bytes', 'dcgm_nvlink_rx_bytes',
                                      'dcgm_pcie_tx_bytes', 'dcgm_pcie_rx_bytes',
                                      'TIMESTAMP', 'START_TIME', 'END_TIME']
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
                                 r'UTIL_PERC|MEM_UTIL_PERC|GFLOPS|gr_engine_active|'
                                 r'sm_active|sm_occupancy|tensor_active|fp64_active|'
                                 r'fp32_active|fp16_active|dram_active|nvlink_tx_bytes|'
                                 r'nvlink_rx_bytes|pcie_tx_bytes|pcie_rx_bytes)')
    gpu_field_map = {'POWER_W': 'gpu_power',
                     'FREQ_KHZ': 'gpu_freq',
                     'MEM_FREQ_KHZ': 'gpu_mem_freq',
                     'UTIL_PERC': 'gpu_util',
                     'MEM_UTIL_PERC': 'gpu_mem_util',
                     'GFLOPS': 'gpu_gflops',
                     'gr_engine_active' : 'dcgm_gr_engine_active',
                     'sm_active' : 'dcgm_sm_active',
                     'sm_occupancy' : 'dcgm_sm_occupancy',
                     'tensor_active' : 'dcgm_pipe_tensor_active',
                     'fp64_active' : 'dcgm_pipe_fp64_active',
                     'fp32_active' : 'dcgm_pipe_fp32_active',
                     'fp16_active' : 'dcgm_pipe_fp16_active',
                     'dram_active' : 'dcgm_dram_active',
                     'nvlink_tx_bytes' : 'dcgm_nvlink_tx_bytes',
                     'nvlink_rx_bytes' : 'dcgm_nvlink_rx_bytes',
                     'pcie_tx_bytes' : 'dcgm_pcie_tx_bytes',
                     'pcie_rx_bytes' : 'dcgm_pcie_rx_bytes'
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
                            r'sm_active|sm_occupancy|tensor_active|fp64_active|'
                            r'fp32_active|fp16_active|dram_active|nvlink_tx_bytes|'
                            r'nvlink_rx_bytes|pcie_tx_bytes|pcie_rx_bytes))'
                            r'|JOBID|STEPID|NODENAME|LOOPID|LOOP_NEST_LEVEL|'
                            r'LOOP_SIZE|TIMESTAMP|START_TIME|END_TIME|time|'
                            r'task_id|app_id|JOBNAME|APPID')
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
                         END_TIME=lambda df: (df.END_TIME -  df_job.START_TIME.min()) * 1000000)
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
        events_info = unique(df_events.Event_type)

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
        file_trace_body = '\n'.join(merge(body_list_sorted,
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
    returned as str. '_<stepid>' region depends on whether `stepid` parameter is not
    None.

    Basic command for each format:
        runtime -> -r -o -> Generates [out_jobs.]tmp_<jobid>[_<stepid>].csv
        ear2prv -> -r -o -> Generates [out_jobs.]tmp_<jobid>[_<stepid>].csv
        summary -> -l -> Generates [events.]tmp_<jobid>[_<stepid>].csv

    If the requested format is "summary" or `ear_events` is True, an
    additional call is done requesting for events, i.e., `eacct -x`.
    The resulting filename is "events.tmp_<jobid>[_<stepid>].csv", but note that
    the function is still returning the basic command filename.
    """

    if stepid is None:
        csv_file = f'tmp_{jobid}.csv'
        job_fmt = f'{jobid}'
    else:
        csv_file = f'tmp_{jobid}_{stepid}.csv'
        job_fmt = f'{jobid}.{stepid}'

    if result_format == 'runtime' or result_format == "ear2prv":
        cmd = ["eacct", "-j", job_fmt, "-r", "-o", "-c", csv_file]
    elif result_format == 'summary':
        cmd = ["eacct", "-j", job_fmt, "-l", "-c", csv_file]
    else:
        print("Unrecognized format: Please contact with support@eas4dc.com")
        exit()

    # Run the command
    res = run(cmd, stdout=PIPE, stderr=PIPE)

    # Check the possible errors
    if "Error getting ear.conf path" in res.stderr.decode('utf-8'):
        print("Error getting ear.conf path")
        exit()

    if "No jobs found" in res.stdout.decode('utf-8'):
        print(f"eacct: {jobid} No jobs found.")
        exit()

    if "No loops retrieved" in res.stdout.decode('utf-8'):
        print("eacct:", job_fmt, "No loops retrieved")
        exit()

    # Request EAR events

    if ear_events or result_format == 'summary':
        cmd = ["eacct", "-j", job_fmt, "-x", '-c',
               '.'.join(['events', csv_file])]
        res = run(cmd, stdout=PIPE, stderr=PIPE)

    if result_format == 'summary':
        output_fn = '.'.join(['loops', csv_file])
        cmd = ["eacct", "-j", job_fmt, "-r", '-o', '-c', output_fn]
        res = run(cmd, stdout=PIPE, stderr=PIPE)

    # Return generated file
    return csv_file


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
        print_configuration(config_file_path)
        return

    print(f'Using {config_file_path} as configuration file...')

    # Show available metrics
    if args.avail_metrics == True:

        comp = function_compose(get_plottable_metrics,
                                read_metrics_configuration)
        config_metrics = comp(config_file_path)
        print(f'Available metrics: {" ".join(config_metrics)}.')
        return

    csv_generated = False

    if args.input_file is None:

        print('This version still requires an input file.'
              'Run an applicatin with --ear-user-db flag.')
        return

        # Action performing eacct command and storing csv files

        input_file = eacct(args.format, args.job_id, args.step_id)

        args.input_file = input_file

        csv_generated = True

    if args.job_id is None:
        print('A Job ID is required for filtering data.')
        return

    if args.format == "runtime" or args.format == "ear2prv":
        head_path, tail_path = path.split(args.input_file)
        out_jobs_path = path.join(head_path,
                                  '.'.join(['out_jobs', tail_path]))

    if args.format == "runtime":

        runtime(args.input_file, out_jobs_path,
                read_metrics_configuration(config_file_path),
                args.metrics, config_file_path, args.manual_range,
                args.title, args.job_id, args.step_id, args.output)

    elif args.format == "ear2prv":
        events_data_path = None
        # if args.events:
        #     events_data_path = (path
        #                         .join(head_path,
        #                               '.'.join(['events', tail_path])))

        # Call ear2prv format method
        ear2prv(out_jobs_path, args.input_file,
                read_job_data_config(config_file_path),
                read_loop_data_config(config_file_path),
                read_events_configuration(config_file_path),
                events_data_fn=events_data_path, job_id=args.job_id,
                step_id=args.step_id, output_fn=args.output)

    elif args.format == 'summary':
        try:
            df_long = (read_data(args.input_file, sep=';')
                       .pipe(filter_df,
                             JOBID=args.job_id,
                             STEPID=args.step_id))
            print(df_long)
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
                    metrics_conf = read_metrics_configuration(config_file_path)
                    phases_conf = read_phases_configuration(config_file_path)

                    build_job_summary(df_long, df_loops, df_events,
                                      metrics_conf, phases_conf)

    if csv_generated and not args.keep_csv:
        system(f'rm {input_file}')
        system(f'rm {out_jobs_path}')
        if args.format == 'ear2prv':
            system(f'rm {out_jobs_path}')
            # if args.events:
            #     system(f'rm {events_data_path}')
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
    parser.add_argument('--version', action='version', version='%(prog)s 5.0')

    main_group = parser.add_argument_group('Main options',
                                           description='''The main option flags
                                           required by the tool.''') 

    main_group.add_argument('-c', '--config-file',
                        help='Specify a custom configuration file.')

    # format and print-config options are mutually exclusive
    main_excl_grp = main_group.add_mutually_exclusive_group(required=True)

    # Specify 
    main_excl_grp.add_argument('--format', choices=['runtime', 'ear2prv', 'summary'],
                               help='''Build results according to chosen format:
                               `runtime` (static images) or `ear2prv` (using paraver
                               tool). `summary` (Beta) option builds a
                               small report about the job metrics.''')

    main_excl_grp.add_argument('--print-config', action='store_true',
                               help='''Prints the used configuration file.''')

    main_excl_grp.add_argument('--avail-metrics', action='store_true',
                               help='''Prints the available metrics provided by
                               the configuration file.''')

    format_grp = parser.add_argument_group('Format common options',
                                           description='''Used when requesting
                                           any of "--format" choices.''')

    format_grp.add_argument('--input-file',
                        help=('''Specifies the input file(s)
                              name(s) to read data from. It can be a path.
                              (Required).'''))

    format_grp.add_argument('-j', '--job-id', type=int,
                            help='Filter the data by the Job ID (Required).')

    format_grp.add_argument('-s', '--step-id', type=int,
                            help='Filter the data by the Step ID.')

    format_grp.add_argument('-o', '--output',
                            help="""Sets the output file name.
                            If a path to an existing directory is given,
                            `runtime` option saves files with the form
                            `runtime_<metric>.pdf` (for each requested metric) will be
                            on the given directory. Otherwise,
                            runtime_<metric>-<output> is stored for each resulting
                            figure.
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

    # ear2prv_group_args = parser.add_argument_group('`ear2prv` format options')

    # events_help_str = 'Include EAR events in the trace fille.'
    # ear2prv_group_args.add_argument('-e', '--events', action='store_true',
    #                                 help=events_help_str)

    return parser


def main():
    """ Entry method. """

    parser = build_parser()

    args = parser.parse_args()

    parser_action(args)


if __name__ == '__main__':
    main()
