######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


"""Methods for generating static images"""
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from . import ear_data as edata
from . import io_api


def start_end_index_1s(start, end, orig_index):
    """Returns an Index based on `orig_index`, starting at `start` and ending
        at `end`. Each entry is 1s from to its previous/next entry.
    """
    return (pd.date_range(start=pd.to_datetime(start, unit='s'),
                          end=pd.to_datetime(end, unit='s'), freq='1s')
            .union(orig_index))


def build_gradient_norm(data_values, step, v_min=None, v_max=None):
    """Return a discretized normalization for `data_values`, maintaining a
        `step` between each interval.
    """
    bounds = np.arange(v_min if v_min is not None else np.nanmin(data_values),
                       v_max + step if v_max is not None
                       else np.nanmax(data_values) + step,
                       step)
    if bounds.size > 16 or bounds.size < 5:
        print(f'Warning! {bounds.size} discrete intervals generated.')

        if bounds.size > 16:
            print(f'Consider increasing the step (currently {step})')
        else:
            print(f'Consider decreasing the step (currently {step})')

    cmap = mpl.colormaps['viridis_r']
    return mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')


def get_elapsed(index, tick_idx):
    """Returns the elapsed time since `index`'s' begining time
    based on `tick_idx`.
    """
    # Compute time deltas to be showed to
    # the end user instead of an absolute timestamp.
    time_deltas = [i - index[0] for i in index]
    return time_deltas[tick_idx].seconds


def generate_metric_timeline_fig(df, app_start_time, app_end_time, metric,
                                 step, **kwargs):
    """
    Generates the timeline gradient figure.

    kwargs:
        - v_min: Heatmap gradient lower bound. Default: None.
        - v_max: Heatmap gradient upper bound. Default: None.
        - fig_title: Resulting figure title. Default: ''.
        - metric_display_name: Specify how the metric name must be displayed.
            Default: ''.
        - gpu_metrics_re: A regex to find GPU columns.

    Returns: A figure.
    """
    m_data = (edata
              .metric_timeseries_by_node(df,
                                         df.filter(regex=metric).columns)
              )

    m_data.index = pd.to_datetime(m_data.index, unit='s')

    m_data = (m_data
              .reindex(start_end_index_1s(app_start_time, app_end_time,
                       m_data.index))
              .bfill())

    m_data_array = m_data.values.transpose()

    # Create the resulting figure for current metric
    print("Creating the figure...")

    fig = plt.figure()
    axs = ImageGrid(fig, 111, nrows_ncols=(len(m_data_array), 1), axes_pad=0,
                    label_mode='L', cbar_mode='single', cbar_location='bottom',
                    cbar_pad=0.5, cbar_size='20%')

    print(f'Setting title: {kwargs.get("fig_title", "")}')
    axs[0].set_title(kwargs.get('fig_title', ''))

    # Normalize values
    norm = build_gradient_norm(m_data_array, step,
                               kwargs.get('v_min', None),
                               kwargs.get('v_max', None))
    gpu_metric_regex = re.compile(kwargs.get('gpu_metrics_re', ''))

    for i, _ in enumerate(m_data_array):
        gpu_metric_match = gpu_metric_regex.search(m_data.columns[i][0])

        if gpu_metric_match:
            ylabel_text = (f'GPU{gpu_metric_match.group(1)}'
                           f' @ {m_data.columns[i][1]}')
        else:
            ylabel_text = m_data.columns[i][1]

        axs[i].grid(axis='x', alpha=0.5)
        axs[i].set_yticks([0], labels=[ylabel_text])
        axs[i].set_xmargin(0)
        axs[i].set_ymargin(0)

        # Generate the timeline gradient
        viridis = mpl.colormaps['viridis_r']
        axs[i].bar(range(len(m_data_array[i])), len(m_data_array[i])/10,
                   width=1, color=[viridis(norm(x)) if not np.isnan(x) else
                                   'white' for x in m_data_array[i]])

    axs[-1].minorticks_on()

    # Create the figure colorbar

    axs.cbar_axes[0].colorbar(mpl.cm.ScalarMappable(norm=norm,
                                                    cmap='viridis_r'),
                              label=kwargs.get('metric_display_name', metric),
                              format=None)

    return fig


def read_runtime_configuration(config_fn):
    """
    Reads the configuration file name passed and returns runtime configuration
    """
    return io_api.read_configuration(config_fn)['runtime']


def runtime_node_metrics_configuration(runtime_config):
    """
    Returns the node metrics configuration from runtime configuration
    """
    return runtime_config['metrics']


def runtime_gpu_metrics_configuration(runtime_config):
    """
    Returns the gpu metrics configuration from runtime configuration
    """
    return runtime_config['gpu_metrics']


def runtime_socket_metrics_configuration(runtime_config):
    """
    Returns the socket metrics configuration from runtime configuration
    """
    return runtime_config['socket_metrics']


def runtime_app_start_time_col(runtime_config):
    """
    Returns the column refering to the START_TIME of an application
    """
    return runtime_config['app_info']['start_time']


def runtime_app_end_time_col(runtime_config):
    """
    Returns the column refering to the START_TIME of an application
    """
    return runtime_config['app_info']['end_time']


def runtime_get_gpu_metrics_regex(runtime_config):
    """
    Returns the regex to match GPU columns
    """
    return runtime_config['gpu_data']['gpu_columns_re']
