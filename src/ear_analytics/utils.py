""" Util functions. """


def filter_df(data_f, **kwargs):
    """
    Filters the DataFrame `data_f`. **kwargs keys indicate the DataFrame
    columns you want to filter by, and keys are values.
    """

    expr = ' and '.join([f'{k} == @kwargs.get("{k}")'
                         for k in kwargs if kwargs[k] is not None
                         and k in data_f.columns])
    if expr == '':
        return data_f

    return data_f.query(expr)


def list_str(values):
    """
    Split the string `values` using comma as a separator.
    """
    return values.split(',')


def join_metric_node(df):
    "Given a DataFrame df, returns it flattening it's columns MultiIndex."
    df.columns = df.columns.to_flat_index()
    return df
