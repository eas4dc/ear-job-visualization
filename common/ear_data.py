def df_get_valid_gpu_data(df):
    """
    Returns a DataFrame with only valid GPU data.

    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """
    gpu_metric_regex_str = (r'GPU(\d)_(POWER_W|FREQ_KHZ|MEM_FREQ_KHZ|'
                            r'UTIL_PERC|MEM_UTIL_PERC)')
    return (df
            .filter(regex=gpu_metric_regex_str)
            .mask(lambda x: x == 0)  # All 0s as nan
            .dropna(axis=1, how='all')  # Drop nan columns
            .mask(lambda x: x.isna(), other=0))  # Return to 0s


def df_has_gpu_data(df):
    """
    Returns whether the DataFrame df has valid GPU data.
    """
    return not df.pipe(df_get_valid_gpu_data).empty
