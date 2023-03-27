from .io_api import read_configuration


def read_events_configuration(filename):
    """
    Returns events configuration stored in `filename`,
    which is a file in JSON format.
    """
    return read_configuration('events')
