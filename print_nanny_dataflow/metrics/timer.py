from time import time
import logging
from functools import wraps
from apache_beam import DoFn
from apache_beam.metrics import Metrics

logger = logging.getLogger(__name__)


def time_distribution(namespace, name):
    return MeasureTimeDecorator(namespace, name)


class MeasureTimeDecorator(object):
    def __call__(self, to_decorate):
        @wraps(to_decorate)
        def decorated(*args, **kwargs):
            start = time()
            result = to_decorate(*args, **kwargs)
            self.time_distribution.update(int((time() - start) * 1000))
            return result

        return decorated

    def __init__(self, namespace, name):
        self.time_distribution = Metrics.distribution(namespace, name)
