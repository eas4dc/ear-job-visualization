""" Implementation of Metric and Metrics classes. """

from matplotlib.colors import Normalize


class Metric:
    """ Manage information of a metric. """

    def __init__(self, key, name=None, values=(0, 1)):
        self.key = key
        self.name = name
        self.range = values

    def set_name(self, name):
        """ Assigns the atribute 'name' of the instance. """
        self.name = name

    def set_values(self, values):
        """ Assigns the range atribute (a tuple of floats) of the instance. """
        self.range = values

    def norm_func(self, clip=True):
        """ Returns the normalization function. """
        return Normalize(vmin=self.range[0], vmax=self.range[1], clip=clip)

    def __str__(self):
        return f'{self.key}: {self.name} -- {self.range}'


class Metrics:
    """ Manage all the metrics that this program can work with. """

    def __init__(self):
        self.metrics = dict()

    def add_metric(self, metric):
        """ Adds a new metric to the control data structure. """
        self.metrics[metric.key] = metric

    def get_metric(self, metric_key):
        """ Returns the metric corresponding to the key passed. """
        return self.metrics.get(metric_key. metric_key)

    def __str__(self):
        res = ''
        for metric in self.metrics.values():
            res += (str(metric) + '\n')
        return res


def init_metrics(config):
    """
    Based on configuration stored in `config`,
    inits the metric types used by this software.
    """
    mts = Metrics()

    for metric in config['METRICS']:
        mts.add_metric(Metric(metric, metric.upper(),
                       config['METRICS'].gettuple(metric)))

    return mts
