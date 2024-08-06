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


def generate_metric_timeline_fig(df, app_start_time, app_end_time, metric, step,
                                 **kwargs):
    """
    Generates the timeline gradient figure.

    kwargs:
        - v_min: Heatmap gradient lower bound. Default: None.
        - v_max: Heatmap gradient upper bound. Default: None.
        - fig_title: Resulting figure title. Default: ''.
        - granularity: Specifies the granularity of the metric (node/app).
            Default: node.
        - metric_display_name: Specify how the metric name must be displayed.
            Default: ''.

    Returns: A figure.

    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """


    granularity = kwargs.get('granularity', 'node')

    if granularity != 'app':
        m_data = edata.metric_timeseries_by_node(df, df.filter(regex=metric).columns)
    else:
        m_data = edata.metric_agg_timeseries(df, df.filter(regex=metric).columns)

    m_data.index = pd.to_datetime(m_data.index, unit='s')

    m_data = (m_data
              .reindex(start_end_index_1s(app_start_time, app_end_time,
                       m_data.index))
              .bfill())

    m_data_array = m_data.values.transpose()

    if granularity == 'app':
        m_data_array = m_data_array.reshape(1, m_data_array.shape[0])

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

    print(f'Setting title: {kwargs.get("fig_title", "")}')
    axs[0].set_title(kwargs.get('fig_title', ''))

    # Normalize values
    norm = build_gradient_norm(m_data_array, step,
                               kwargs.get('v_min', None),
                               kwargs.get('v_max', None))

    gpu_metric_regex = re.compile((r'GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                                   r'UTIL_PERC|MEM_UTIL_PERC|'
                                   r'(10[01][0-9]))'))

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

        # Generate the timeline gradient
        ax.imshow(np.array(m_data_array[i], ndmin=2), norm=norm,
                  cmap=mpl.colormaps['viridis_r'],
                  interpolation='none', aspect=2*len(m_data_array))

    def format_fn(tick_val):
        """
        Map each tick with the corresponding
        elapsed time to label the timeline.
        """
        return get_elapsed(m_data.index, tick_val)

    xticks = np.arange(len(m_data_array[0]), step=20)
    axs[-1].set(xticks=xticks,
                xticklabels=map(format_fn, xticks))
    axs[-1].minorticks_on()

    # Create the figure colorbar

    axs.cbar_axes[0].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
                              kwargs.get('metric_display_name', metric),
                              format=None)
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r'),
    #              cax=cb[0], location='bottom', label=label, format='%.2f')

    return fig
