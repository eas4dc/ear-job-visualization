from pandas import concat, DataFrame
from numpy import around
from re import findall
from pylatex import Tabular
from pylatex.utils import bold

from .metrics import metric_regex, read_metrics_configuration
from .ear_data import df_has_gpu_data, df_get_valid_gpu_data


def job_cpu_summary_df(df, metrics_conf):
    """
    Given a DataFrame df representing raw EAR data in long format,
    i.e., eacct -l, returns a DataFrame with a summary of relevant metrics
    aggregated.
    """

    # TODO: Pay attention here because this function depends directly
    # on EAR's output.

    grouped = (df
               .assign(
                   # Energy column
                   energy=lambda x: (x[metric_regex('dc_power', metrics_conf)]\
                                     * x[metric_regex('time_sec',
                                         metrics_conf)]),
                   # gflops_w=lambda x: (x['CPU-GFLOPS'] /
                   #                     x['DC_NODE_POWER_W'])
               )
               .groupby(['JOBID', 'STEPID']))

    grouped_n_nodes = grouped[[metric_regex('node_count',
                                            metrics_conf)]].count()

    metrics_to_mean = ['time_sec', 'cpi', 'perc_mpi']

    metrics_to_mean_re_lst = [metric_regex(metric, metrics_conf) for metric
                              in metrics_to_mean]

    grouped_avg_metrics = grouped[metrics_to_mean_re_lst].mean()

    freqs_metrics = ['avg_cpufreq', 'avg_imcfreq', 'def_freq']
    freqs_metrics_re_lst = [metric_regex(metric, metrics_conf) for metric
                            in freqs_metrics]

    # Frequencies need to be passed to GHz
    grouped_freqs_metrics = (grouped[freqs_metrics_re_lst]
                             .mean().div(pow(10, 6)))

    metrics_to_agg = ['dc_power', 'energy', 'cpu_gflops', 'gbs', 'io_mbs']
    metrics_to_agg_re_lst = [metric_regex(metric, metrics_conf) for metric
                             in metrics_to_agg]
    grouped_agg_metrics = grouped[metrics_to_agg_re_lst].sum()

    field_to_str = {
            'job_step': 'Job-Step',
            'NODENAME': '# Nodes',  # NODENAME field was recycled,
                                    # but count was used.
            'AVG_CPUFREQ_KHZ': 'Average CPU freq. (GHz)',
            'AVG_IMCFREQ_KHZ': 'Average IMC freq. (GHz)',
            'TIME_SEC': 'Average execution time (s)',
            'DC_NODE_POWER_W': 'Accumulated node power (W)',
            'energy': 'Total energy (J)',
            'CPU-GFLOPS': 'Accumulated GFlop/s',
            'MEM_GBS': 'Accumulated Mem. bandwidth (GB/s)',
            'IO_MBS': 'Accumulated I/O (MB/s)',
            'PERC_MPI': 'Average % MPI',
            'DEF_FREQ_KHZ': 'Default freq. (GHz)'
            }

    return (concat([grouped_n_nodes, grouped_avg_metrics,
                    grouped_freqs_metrics, grouped_agg_metrics], axis=1)
            .transform(around, decimals=2)
            .reset_index()
            .assign(job_step=lambda x: (f'{x.JOBID.values[0]}-'
                                        f'{x.STEPID.values[0]}'))
            .pipe(lambda df: df[['job_step',
                                 'NODENAME',
                                 'DEF_FREQ_KHZ',
                                 'AVG_CPUFREQ_KHZ',
                                 'AVG_IMCFREQ_KHZ',
                                 'TIME_SEC',
                                 'DC_NODE_POWER_W',
                                 'energy',
                                 # 'gflops_w',
                                 'MEM_GBS',
                                 'CPI',
                                 'IO_MBS',
                                 'PERC_MPI']])
            .transpose()
            .rename(index=field_to_str))


def job_gpu_summary(df, conf_fn='config.json'):

    metrics_conf = read_metrics_configuration(conf_fn)
    gpu_pwr_regex = metric_regex('gpu_power', metrics_conf)

    if df_has_gpu_data(df):

        df_gpu_data = (df_get_valid_gpu_data(df)
                       .assign(
                           total_gpu_power=lambda x: (x
                                                      .filter(regex=gpu_pwr_regex)
                                                      .sum(axis=1))
                           ))

        df_gpus_used = df_gpu_data.filter(regex=metric_regex('gpu_util',
                                                             metrics_conf))
        gpus_used = findall(r'\d', ''.join(df_gpus_used.columns))

        gpus_used_id_re = '|'.join(gpus_used)
        power_used_re = f'GPU({gpus_used_id_re})_POWER_W'

        field_to_str = {
                'total_gpu_power': 'Total GPU Power (W)',
                'used_gpu_power': 'Total (used) GPU Power (W)'
                }

        return (df_gpu_data
                .assign(used_gpu_power=lambda x: (x
                                                  .filter(regex=power_used_re)
                                                  .sum(axis=1)
                                                  )
                        )
                .pipe(lambda df: df[['total_gpu_power', 'used_gpu_power']])
                .rename(field_to_str, axis=1)
                .sum()
                .transform(around, decimals=2)
                .pipe(lambda series: DataFrame({0: series}))
                )


def job_summary_to_tex_tabular(df, filepath, **kwargs):
    """
    Prints the DataFrame `df` to `filepath`.tex file in LaTeX format.
    The DataFrame must to have the shape (_, 2), i.e., two columns only.
    kwargs are passed to the Tabular class creation.
    """

    tabular_params = '|r|l|'

    tabular = Tabular(tabular_params, **kwargs)

    df_values = list(df.index.map(lambda x: df.loc[x][0]))

    tabular.add_hline()

    for field, value in zip(df.index, df_values):
        tabular.add_row(bold(field), value)
        tabular.add_hline()

    tabular.generate_tex(filepath)
