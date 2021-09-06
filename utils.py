""" Util functions. """

import pandas as pd


def filter_by_job_step_app(data_f, job_id=None, step_id=None, app_id=None):
    """
    Filters the DataFrame `data_f` by `job_id`
    and/or `step_id` and/or `app_id`.
    """

    def mask(data_f, key, value):
        if key in data_f.columns:
            if value is not None:
                return data_f[data_f[key] == value]
        # print(f'{key} is not a column name')
        return data_f

    pd.DataFrame.mask = mask

    return (data_f
            .mask('APP_ID', app_id)
            .mask('JOB_ID', job_id)
            .mask('JID', job_id)
            .mask('STEPID', step_id)
            )
