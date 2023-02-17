from .io_api import read_configuration


def read_metrics_configuration(filename):
    """
    Return metrics configuration stored in `filename`,
    which is a file in JSON format.
    """
    return read_configuration(filename)['metrics']


def metric_regex(metric, metrics_conf):
    """
    This function returns the metric's column name
    regex to be used then in a filtering action .
    """

    return metrics_conf[metric]['column_name']
